# Fix 1: Modify RJGPModel to store the correct permutation
struct RJGPModel
    grid::FUBARgrid
    Σ::Matrix{Float64} # Full kernel
    dimension::Int64
    purifying_prior::Float64
    ess_to_fubar_perm::Vector{Int64}  # Changed name to be explicit
end

"""
    generate_ess_indices(N::Int)

Generate ESS format indices for an NxN grid. Indices are generated along diagonals
from bottom-right to top-left, with each diagonal numbered from bottom to top.
"""
function generate_ess_indices(N::Int)
    indices = zeros(Int, N, N)
    current_index = 1

    # Loop over each diagonal, starting from bottom right
    for diag in (2N-2):-1:0
        # Calculate range for this diagonal
        col_start = max(0, diag - N + 1)
        col_end = min(diag, N - 1)

        # Store positions for this diagonal (bottom to top)
        positions = [(diag - col, col)
                     for col in col_end:-1:col_start]

        # Fill in indices
        for pos in positions
            indices[pos[1]+1, pos[2]+1] = current_index
            current_index += 1
        end
    end

    return indices
end

"""
    generate_fubar_indices(N::Int)

Generate FUBAR format indices for an NxN grid. Indices are generated column-wise
from bottom to top, starting from the leftmost column.
"""
function generate_fubar_indices(N::Int)
    indices = zeros(Int, N, N)
    current_index = 1

    # Fill column by column, bottom to top
    for col in 1:N
        for row in N:-1:1
            indices[row, col] = current_index
            current_index += 1
        end
    end

    return indices
end

function get_ess_to_fubar_permutation(N::Int)
    ess = generate_ess_indices(N)
    fubar = generate_fubar_indices(N)

    # Create mapping from ESS indices to FUBAR indices
    n_elements = N * N
    perm = zeros(Int, n_elements)

    for i in 1:N
        for j in 1:N
            ess_idx = ess[i, j]
            fubar_idx = fubar[i, j]
            # We want to map FROM ess TO fubar, so:
            perm[fubar_idx] = ess_idx  # This line was wrong before!
        end
    end

    return perm
end

function get_fubar_to_ess_permutation(N::Int)
    ess = generate_ess_indices(N)
    fubar = generate_fubar_indices(N)

    # Create mapping from FUBAR indices to ESS indices
    n_elements = N * N
    perm = zeros(Int, n_elements)

    for i in 1:N
        for j in 1:N
            ess_idx = ess[i, j]
            fubar_idx = fubar[i, j]
            # We want to map FROM fubar TO ess, so:
            perm[ess_idx] = fubar_idx  # This line was wrong before!
        end
    end

    return perm
end

# Fix 3: Update the loglikelihood function
function loglikelihood(model::RJGPModel, θ)
    full_θ = [softmax(θ); zeros(model.dimension - length(θ))]
    return sum(log.(full_θ[model.ess_to_fubar_perm]'model.grid.cond_lik_matrix))
end


function rearrange_kernel_matrix(Σ)
    fubar_to_ess = get_fubar_to_ess_permutation(Int64(sqrt(size(Σ)[1])))
    return Σ[fubar_to_ess, fubar_to_ess]
end


function gaussian_kernel_matrix(grid; kernel_scaling=1.0)
    return kernel_matrix(grid, distance_function=x -> exp(-x / (2 * kernel_scaling^2)))
end

function kernel_matrix(grid; distance_function=x -> exp(-x))
    n_points = length(grid.alpha_ind_vec)
    distances = zeros(eltype(distance_function(0.0)), n_points, n_points)
    for i in 1:n_points
        for j in 1:n_points
            distance = (grid.alpha_ind_vec[i] - grid.alpha_ind_vec[j])^2 +
                       (grid.beta_ind_vec[i] - grid.beta_ind_vec[j])^2
            distances[i, j] = distance_function(distance)
        end
    end
    return distances
end

# Optimize kernel_matrix_c for better performance
function kernel_matrix_c(grid::FUBARgrid, c::Real)
    n_points = length(grid.alpha_ind_vec)
    K = zeros(eltype(grid.grid_values), n_points, n_points)
    inv_2c_squared = 1.0 / (2 * c^2)  # Precompute this value

    @inbounds for i in 1:n_points
        for j in 1:n_points
            # Calculate distance using alpha_ind_vec and beta_ind_vec like in kernel_matrix
            distance = (grid.alpha_ind_vec[i] - grid.alpha_ind_vec[j])^2 +
                       (grid.beta_ind_vec[i] - grid.beta_ind_vec[j])^2
            K[i, j] = exp(-distance * inv_2c_squared)
        end
    end
    return K
end

# Then define the adjoint for automatic differentiation
Zygote.@adjoint function kernel_matrix_c(grid::FUBARgrid, c::Real)
    n_points = length(grid.alpha_ind_vec)
    K = zeros(eltype(grid.grid_values), n_points, n_points)
    D = zeros(eltype(grid.grid_values), n_points, n_points)  # store distances for the backward pass

    for i in 1:n_points
        for j in 1:n_points
            # Calculate distance using alpha_ind_vec and beta_ind_vec
            distance = (grid.alpha_ind_vec[i] - grid.alpha_ind_vec[j])^2 +
                       (grid.beta_ind_vec[i] - grid.beta_ind_vec[j])^2
            D[i, j] = distance
            K[i, j] = exp(-distance / (2 * c^2))
        end
    end

    function kernel_matrix_c_pullback(dK)
        # Instead of using zero(grid), we'll return nothing for the grid gradient
        # since we're not differentiating with respect to the grid
        grad_c = zero(c)
        # For each element, accumulate the contribution to the derivative with respect to c.
        # d/dc exp(-d/(2*c^2)) = exp(-d/(2*c^2)) * (d / c^3)
        for i in 1:n_points
            for j in 1:n_points
                grad_c += dK[i, j] * K[i, j] * (D[i, j] / c^3)
            end
        end
        return (nothing, grad_c)
    end

    return K, kernel_matrix_c_pullback
end

# Fix 2: Update the model constructor
function generate_RJGP_model(grid::FUBARgrid; kernel_scaling=1.0, purifying_prior=1 / 2)
    Σ = rearrange_kernel_matrix(gaussian_kernel_matrix(grid, kernel_scaling=kernel_scaling))
    dimension = size(Σ)[1]
    return RJGPModel(
        grid,
        Σ,
        dimension,
        purifying_prior,
        get_ess_to_fubar_permutation(Int64(sqrt(dimension)))  # Changed to ESS→FUBAR
    )
end
function rjess(model::RJGPModel; ϵ=0.01, n_samples=1000, model_switching_probability=0.3, prior_only=false, diagnostics=false)
    ll = prior_only ? x -> 0 : x -> loglikelihood(model, x)
    full_covariance = 1 / 2 * (model.Σ + model.Σ') + (ϵ * I) # Tikhonov + posdef, need big ϵ for high kernel scaling
    N = Int64(sqrt(model.dimension))
    smallest_model_dimension = Int64(N * (N - 1) / 2)
    model_dimensions = accumulate((x, i) -> x + (N - i), 0:(N-1); init=smallest_model_dimension)
    # Uniform prior for positive selection models, "lump" prior for purifying
    diversifying_model_prior = (1 - model.purifying_prior) / (length(model_dimensions) - 1)
    model_priors = [[model.purifying_prior]; diversifying_model_prior .* ones(length(model_dimensions) - 1)]
    if diagnostics
        println("Model dimension: ", model_dimensions[end])
        println("Σ minimum eigenvalue: ", minimum(eigvals(full_covariance)))
        println("Model priors: ", model_priors)
        println("Model dimensions: ", model_dimensions)
    end
    problem = generate_reversible_slice_sampler(full_covariance, model_dimensions, model_priors, ll; prior_only=prior_only)

    return reversible_slice_sampling(problem, n_samples=n_samples, jump_proposal_probability=model_switching_probability)
end

function plot_logposteriors_with_transitions(model_indices, logposteriors)
    # Initialize empty plot
    p = plot(legend=true)

    # Find transition points
    transitions = findall(diff(model_indices) .!= 0)

    # Add first segment
    start_idx = 1

    for transition in transitions
        # Plot segment up to transition
        plot!(p, start_idx:transition, logposteriors[start_idx:transition],
            linewidth=2, label=(start_idx == 1 ? "Log Posterior" : false))

        # Add vertical red line at transition
        vline!([transition], color=:red,
            linestyle=:solid, label=(start_idx == 1 ? "Model Transition" : false))

        # Start next segment after transition
        start_idx = transition + 1
    end

    # Plot final segment
    if start_idx <= length(logposteriors)
        plot!(p, start_idx:length(logposteriors),
            logposteriors[start_idx:end],
            linewidth=2, label=false)
    end

    return p
end

function compute_rjess_to_fubar_permutation(v::AbstractVector{Float64}, N::Int)
    ess_to_fubar = get_ess_to_fubar_permutation(N)  # Get ESS→FUBAR permutation
    return v[ess_to_fubar]  # Apply it to transform ESS→FUBAR
end


function softmax(x)
    exp_x = exp.(x .- maximum(x))
    return exp_x ./ sum(exp_x)
end

function format_sample(sample, max_dimension)
    return [softmax(sample); zeros(max_dimension - length(sample))]
end

function compute_purifying_bayes_factor(model_indices, purifying_prior)
    # Calculate burnin (20% of samples)
    burnin = Int(floor(0.2 * length(model_indices)))

    # Use only post-burnin samples
    post_burnin_indices = model_indices[(burnin+1):end]

    # Count visits to model 1 (purifying) and models > 1 (non-purifying)
    purifying_visits = count(x -> x == 1, post_burnin_indices)
    non_purifying_visits = count(x -> x > 1, post_burnin_indices)

    # Compute posterior odds
    posterior_odds = purifying_visits / non_purifying_visits

    # Compute prior odds
    prior_odds = purifying_prior / (1 - purifying_prior)

    # Bayes factor = posterior odds / prior odds
    bayes_factor = posterior_odds / prior_odds

    return bayes_factor
end

function gpFUBAR(problem::RJGPModel; ϵ=0.01, n_samples=1000, model_switching_probability=0.01, prior_only=false, diagnostics=false)
    samples, model_indices, logposteriors = rjess(problem, ϵ=ϵ, n_samples=n_samples, model_switching_probability=model_switching_probability, prior_only=prior_only)
    println("Model indices: ", model_indices[1:20])
    println("Logposteriors: ", logposteriors[1:20])
    println("Samples: ", samples[1])
    # Use all samples for Bayes factor calculation and model frequencies
    bayes_factor = 1 / compute_purifying_bayes_factor(model_indices, problem.purifying_prior)

    # Calculate and print model time distributions using all samples
    model_counts = countmap(model_indices)
    total_samples = length(model_indices)
    # println("Jump acceptance rate: ", jump_diagnostics.accepted / jump_diagnostics.proposed)
    if diagnostics
        println("\nModel time distributions:")
    end
    for (model, count) in sort(collect(model_counts))
        percentage = (count / total_samples) * 100
        if diagnostics
            println("Model $model: $(round(percentage, digits=2))% ($(count) samples)")
        end
    end
    println("\nBayes factor (M1/M>1): ", bayes_factor)

    formatted_samples = [format_sample(sample, problem.dimension) for sample in samples]
    fubar_samples = [compute_rjess_to_fubar_permutation(formatted_sample, Int64(sqrt(problem.dimension))) for formatted_sample in formatted_samples]

    posterior_mean = mean(fubar_samples[1:100:end]) # Only plot every 1000th sample
    active_parameters = length.(samples[1:100:end]) # Only plot every 1000th sample

    # Create individual plots
    posterior_mean_plot = gridplot(problem.grid.alpha_ind_vec, problem.grid.beta_ind_vec,
        problem.grid.grid_values, posterior_mean,
        title="Posterior Mean")
    active_parameter_trace = plot(active_parameters,
        title="Active Parameters",
        xlabel="Iteration",
        ylabel="Number of Parameters")
    logposterior_plot = plot_logposteriors_with_transitions(model_indices[1:100:end], logposteriors[1:100:end])

    # Create diagnostic plots layout with modified dimensions
    diagnostic_plots = plot(active_parameter_trace, logposterior_plot,
        layout=(2, 1),
        size=(1000, 800))  # Changed from (600, 1800) to (800, 400)

    # Create animation
    anim = @animate for i in 1:1000:length(formatted_samples)
        gridplot(problem.grid.alpha_ind_vec, problem.grid.beta_ind_vec, problem.grid.grid_values, fubar_samples[i])
    end
    return posterior_mean_plot, diagnostic_plots, anim
end


function quintic_smooth_transition(x, alpha, beta)
    if x <= alpha
        return 0.0
    elseif x >= beta
        return 1.0
    else
        # Normalize x to [0,1] range
        t = (x - alpha) / (beta - alpha)
        # Quintic polynomial that has zero derivatives at t=0 and t=1
        # and exactly reaches 0 at t=0 and 1 at t=1 
        return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
    end
end

# Example smoothing function (logistic) and its derivative.
function smoothing_function(y, α, β)
    return 1 / (1 + exp(-α * (y - β)))
end

function smoothing_function_prime(y, α, β)
    s = exp(-α * (y - β))
    return α * s / (1 + s)^2
end

# Optimize supress_vector for better performance
function supress_vector(θ, αs, βs, dimensions, smoothing_function)
    T = eltype(θ)
    grid_θ = @view θ[1:end-1]  # Use view instead of copying
    y = θ[end]

    s = softmax(grid_θ)
    M = ones(T, length(s))

    @inbounds for i in 2:length(dimensions)
        start_index = dimensions[i-1] + 1
        end_index = dimensions[i]
        f_i = smoothing_function(y, αs[i], βs[i])
        M[start_index:end_index] .= f_i
    end

    A = s .* M
    T_sum = sum(A)
    return A ./ T_sum
end

# Custom adjoint for supress_vector.
Zygote.@adjoint function supress_vector(θ, αs, βs, dimensions, smoothing_function)
    T = eltype(θ)
    grid_θ = θ[1:end-1]
    y = θ[end]

    s = softmax(grid_θ)
    M = ones(T, length(s))
    for i in 2:length(dimensions)
        start_index = dimensions[i-1] + 1
        end_index = dimensions[i]
        f_i = smoothing_function(y, αs[i], βs[i])
        M[start_index:end_index] .= f_i
    end
    A = s .* M
    T_sum = sum(A)
    z = A ./ T_sum

    function pullback(dz)
        # Propagate through the normalization: z = A / T_sum.
        sum_dz = sum(dz)
        dA = dz ./ T_sum .- (sum_dz / T_sum^2) .* A

        # Split the gradient: A = s .* M.
        ds = dA .* M
        dM = dA .* s

        # Compute gradient for softmax input:
        dot_ds_s = sum(ds .* s)
        dgrid = s .* (ds .- dot_ds_s)

        # Compute gradient with respect to y via dM, affecting only groups 2:end.
        dy = zero(y)
        for i in 2:length(dimensions)
            start_index = dimensions[i-1] + 1
            end_index = dimensions[i]
            fprime = smoothing_function_prime(y, αs[i], βs[i])
            dy += fprime * sum(dM[start_index:end_index])
        end

        dθ = similar(θ)
        dθ[1:end-1] .= dgrid
        dθ[end] = dy

        # Return gradients for all arguments: only θ is differentiated.
        return (dθ, nothing, nothing, nothing, nothing)
    end

    return z, pullback
end


function supression_loglikelihood(model::RJGPModel, θ, αs, βs, dimensions)
    T = eltype(θ)
    softmax_grid_θ = supress_vector(θ, αs, βs, dimensions, quintic_smooth_transition)
    return sum(log.(softmax_grid_θ[model.ess_to_fubar_perm]'model.grid.cond_lik_matrix))
end

function non_rj_gpFUBAR(problem::RJGPModel; suppression_σ=1, n_samples=1000, prior_only=false, diagnostics=false)
    new_Σ = zeros(problem.dimension + 1, problem.dimension + 1)
    new_Σ[1:problem.dimension, 1:problem.dimension] = problem.Σ
    new_Σ[problem.dimension+1, problem.dimension+1] = suppression_σ^2
    dimensions = accumulate((x, i) -> x + (20 - i), 0:(20-1); init=190) #Hard coded 20,190 for now
    αs = 0.1 .* [i for i in 0:(length(dimensions)-1)]
    βs = 0.1 .* [i for i in 1:(length(dimensions))]

    actual_loglikelihood = prior_only ? x -> 0 : x -> supression_loglikelihood(problem, x, αs, βs, dimensions)

    ESS_model = ESSModel(MvNormal(zeros(problem.dimension + 1), new_Σ), actual_loglikelihood)
    samples = AbstractMCMC.sample(ESS_model, ESS(), n_samples, progress=true)



    grid_samples = [compute_rjess_to_fubar_permutation(supress_vector(samples[i], αs, βs, dimensions, quintic_smooth_transition), Int64(sqrt(problem.dimension))) for i in eachindex(samples)]

    supression_parameter_samples = [samples[i][end] for i in eachindex(samples)]
    posterior_mean = mean(grid_samples[1:100:end]) # Only plot every 1000th sample

    # Create individual plots
    posterior_mean_plot = gridplot(problem.grid.alpha_ind_vec, problem.grid.beta_ind_vec,
        problem.grid.grid_values, posterior_mean,
        title="Posterior Mean")
    supression_parameter_trace = plot(supression_parameter_samples[1:100:end],
        title="Supression Parameter",
        xlabel="Iteration",
        ylabel="Supression Parameter")
    println("Bayes factor: ", sum(supression_parameter_samples .> 0.0) / sum(supression_parameter_samples .< 0.0))

    anim = @animate for i in 1:1000

        sample = compute_rjess_to_fubar_permutation(supress_vector(samples[i], αs, βs, dimensions, quintic_smooth_transition), Int64(sqrt(problem.dimension)))

        gridplot(problem.grid.alpha_ind_vec, problem.grid.beta_ind_vec, problem.grid.grid_values, sample, title="Supression Parameter: $(i/500)")
    end


    return posterior_mean_plot, supression_parameter_trace, anim
end

# We will solve this with HMC

struct LogPosteriorRJGP
    model::RJGPModel
    αs::Vector{Float64}
    βs::Vector{Float64}
    dimensions::Vector{Int64}
    c_prior::Distribution
    suppression_σ::Float64
    fubar_to_ess_perm::Vector{Int64}
    # Add caching fields
    last_c::Ref{Float64}
    cached_kernel::Ref{Matrix{Float64}}
    cached_cholesky::Ref{Any}  # Will store Cholesky decomposition
end

function initialise_log_posterior_rjgp(model::RJGPModel, c_prior::Distribution, suppression_σ::Float64)
    dimensions = accumulate((x, i) -> x + (20 - i), 0:(20-1); init=190) #Hard coded 20,190 for now
    αs = 0.1 .* [i for i in 0:(length(dimensions)-1)]
    βs = 0.1 .* [i for i in 1:(length(dimensions))]
    # Pre-compute the permutation
    fubar_to_ess_perm = get_fubar_to_ess_permutation(Int64(sqrt(model.dimension)))

    # Initialize with NaN to ensure first calculation
    initial_c = -1.0
    initial_kernel = zeros(model.dimension, model.dimension)

    return LogPosteriorRJGP(
        model, αs, βs, dimensions, c_prior, suppression_σ, fubar_to_ess_perm,
        Ref(initial_c), Ref(initial_kernel), Ref{Any}(nothing)
    )
end

# Optimized log density function
function (problem::LogPosteriorRJGP)(θ)
    c = θ[end]
    non_kernel_θ = @view θ[1:(end-1)]

    # Calculate log likelihood
    log_likelihood = supression_loglikelihood(problem.model, non_kernel_θ, problem.αs, problem.βs, problem.dimensions)

    # Only recompute kernel if c has changed
    if c != problem.last_c[]
        # Generate the kernel matrix with the current c value
        raw_kernel_Σ = kernel_matrix_c(problem.model.grid, c)

        # Apply the permutation
        kernel_Σ = raw_kernel_Σ[problem.fubar_to_ess_perm, problem.fubar_to_ess_perm]

        # Add regularization
        ϵ = 1e-6
        kernel_Σ += ϵ * I

        # Cache the kernel and its Cholesky decomposition
        problem.cached_kernel[] = kernel_Σ
        problem.cached_cholesky[] = cholesky(Symmetric(kernel_Σ))
        problem.last_c[] = c
    end

    # Use cached Cholesky for log density calculation
    c_log_prior = logpdf(problem.c_prior, c)

    # Use the cached Cholesky decomposition for faster MvNormal logpdf
    chol = problem.cached_cholesky[]
    non_kernel_log_prior = -0.5 * (
        problem.model.dimension * log(2π) +
        logdet(chol) +
        dot(non_kernel_θ[1:(end-1)], chol \ non_kernel_θ[1:(end-1)])
    ) + logpdf(Normal(0, problem.suppression_σ), non_kernel_θ[end])

    return log_likelihood + c_log_prior + non_kernel_log_prior
end

LogDensityProblems.logdensity(problem::LogPosteriorRJGP, θ) = problem(θ)
LogDensityProblems.dimension(problem::LogPosteriorRJGP) = problem.model.dimension + 2
LogDensityProblems.capabilities(::Type{LogPosteriorRJGP}) = LogDensityProblems.LogDensityOrder{0}()



function kernel_sampling_non_rj_gpFUBAR(problem::RJGPModel; n_samples=1000, prior_only=false, diagnostics=false,
    sample_c=true, c_step_size=0.01, c_initial=2.0, supression_σ=1.0)
    # Create the log posterior problem instance
    log_posterior = initialise_log_posterior_rjgp(problem, Normal(2, 0.25), supression_σ)

    # Create the AD gradient model
    ad_model = LogDensityProblemsAD.ADgradient(Val(:Zygote), log_posterior)

    model = AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(Val(:Zygote), log_posterior))
    δ = 0.8
    sampler = NUTS(δ)
    AdvancedHMC.ProgressMeter.ijulia_behavior(:clear)
    samples = AbstractMCMC.sample(
        model,
        sampler,
        n_samples;
        initial_params=randn(problem.dimension + 2),
    )
    θ_samples = [samples[i].z.θ for i in eachindex(samples)]
    c_samples = [θ_samples[i][end] for i in eachindex(θ_samples)]
    supression_samples = [θ_samples[i][end-1] for i in eachindex(θ_samples)]
    
    # Calculate log likelihood for each sample
    log_likelihoods = Float64[]
    
    # Define dimensions, αs, and βs for suppression vector calculation
    dimensions = accumulate((x, i) -> x + (20 - i), 0:(20-1); init=190) #Hard coded 20,190 for now
    αs = 0.1 .* [i for i in 0:(length(dimensions)-1)]
    βs = 0.1 .* [i for i in 1:(length(dimensions))]

    # Process samples to get grid values in FUBAR format
    grid_θ_samples = [θ_samples[i][1:(end-2)] for i in eachindex(θ_samples)]
    
    # Calculate suppressed vectors and convert to FUBAR format
    N = Int64(sqrt(problem.dimension))
    fubar_samples = []
    for i in eachindex(grid_θ_samples)
        # Create full parameter vector with suppression parameter
        full_params = [grid_θ_samples[i]; supression_samples[i]]
        # Calculate suppressed vector
        suppressed = supress_vector(full_params, αs, βs, dimensions, quintic_smooth_transition)
        # Convert to FUBAR format
        fubar_sample = compute_rjess_to_fubar_permutation(suppressed, N)
        push!(fubar_samples, fubar_sample)
        
        # Calculate log likelihood for this sample
        ll = supression_loglikelihood(problem, full_params, αs, βs, dimensions)
        push!(log_likelihoods, ll)
    end
    
    # Calculate posterior mean using all samples
    posterior_mean = mean(fubar_samples)
    
    # Create plots
    
    # 1. Posterior mean plot
    posterior_mean_plot = gridplot(
        problem.grid.alpha_ind_vec, 
        problem.grid.beta_ind_vec,
        problem.grid.grid_values, 
        posterior_mean,
        title="Posterior Mean"
    )
    
    # 2. Parameter trace plots
    c_trace_plot = plot(
        c_samples,
        title="Kernel Parameter (c)",
        xlabel="Iteration",
        ylabel="Value",
        legend=false
    )
    
    suppression_trace_plot = plot(
        supression_samples,
        title="Suppression Parameter",
        xlabel="Iteration",
        ylabel="Value",
        legend=false
    )
    
    # 3. Log likelihood plot
    log_likelihood_plot = plot(
        log_likelihoods,
        title="Log Likelihood",
        xlabel="Iteration",
        ylabel="Log Likelihood",
        legend=false
    )
    
    # Combine trace plots
    trace_plots = plot(
        c_trace_plot, 
        suppression_trace_plot,
        log_likelihood_plot,
        layout=(3, 1),
        size=(800, 900)
    )
    
    # 4. Create animation of samples
    # Use all samples for animation (or a reasonable number if there are too many)
    max_frames = min(n_samples, 100)  # Cap at 100 frames for reasonable file size
    frame_indices = n_samples <= max_frames ? (1:n_samples) : round.(Int, range(1, n_samples, length=max_frames))
    
    anim = @animate for i in frame_indices
        gridplot(
            problem.grid.alpha_ind_vec, 
            problem.grid.beta_ind_vec, 
            problem.grid.grid_values, 
            fubar_samples[i],
            title="Sample $i: c=$(round(c_samples[i], digits=2)), s=$(round(supression_samples[i], digits=2))"
        )
    end
    
    # 5. Calculate Bayes factor based on suppression parameter
    # Positive suppression parameter indicates evidence for selection
    positive_count = sum(supression_samples .> 0.0)
    negative_count = sum(supression_samples .< 0.0)
    bayes_factor = positive_count / max(1, negative_count)  # Avoid division by zero
    
    if diagnostics
        println("Kernel parameter (c) mean: ", mean(c_samples))
        println("Suppression parameter mean: ", mean(supression_samples))
        println("Bayes factor (positive/negative): ", bayes_factor)
    end
    
    return posterior_mean_plot, trace_plots, anim, (c_samples=c_samples, suppression_samples=supression_samples, log_likelihoods=log_likelihoods)
end