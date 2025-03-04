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

function rearrange_kernel_matrix(Σ)
    fubar_to_ess = get_fubar_to_ess_permutation(Int64(sqrt(size(Σ)[1])))
    return Σ[fubar_to_ess, fubar_to_ess]
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

# Optimize supress_vector for better performance
function supress_vector(θ, αs, βs, dimensions)
    T = eltype(θ)
    grid_θ = @view θ[1:end-1]  # Use view instead of copying
    y = θ[end]

    s = softmax(grid_θ)
    M = ones(T, length(s))

    @inbounds for i in 2:length(dimensions)
        start_index = dimensions[i-1] + 1
        end_index = dimensions[i]
        f_i = quintic_smooth_transition(y, αs[i], βs[i])
        M[start_index:end_index] .= f_i
    end

    A = s .* M
    T_sum = sum(A)
    return A ./ T_sum
end

function benjamin_loglikelihood(model::RJGPModel, αs, βs, dimensions, fubar_to_ess, cholesky_factor, θ; ϵ = 1e-2) 
    # Extract parameters
    grid_θ = @view θ[1:(end-2)]  # Use view instead of copying
    y = θ[end-1]  # Suppression parameter
    c = exp(θ[end] / 2)  # Kernel scaling factor
    
    # Compute kernel matrix and regularize
    Σ = kernel_matrix_c(model.grid, c)
    Σ = Σ[fubar_to_ess, fubar_to_ess]
    regularised_Σ = Symmetric(Σ) + ϵ * I  # Use Symmetric for better performance
    
    # Transform parameters
    L = cholesky(regularised_Σ).L
    transformed_grid_θ = zeros(eltype(θ), length(grid_θ) + 1)
    transformed_grid_θ[1:(end-1)] = L * grid_θ  # Revert to standard multiplication
    transformed_grid_θ[end] = y
    
    # Apply suppression and compute log-likelihood
    supressed_grid_θ = supress_vector(transformed_grid_θ, αs, βs, dimensions)
    
    # Keep the original matrix multiplication logic
    return sum(log.(supressed_grid_θ[model.ess_to_fubar_perm]' * model.grid.cond_lik_matrix))
end

function ess_benjamin_trick(problem::RJGPModel; n_samples = 1000, prior_only = false)
    fubar_to_ess = get_fubar_to_ess_permutation(20)
    dimensions = accumulate((x, i) -> x + (20 - i), 0:(20-1); init=190) #Hard coded 20,190 for now
    αs = 0.1 .* [i for i in 0:(length(dimensions)-1)]
    βs = 0.1 .* [i for i in 1:(length(dimensions))]
    # cholesky_factor = cholesky(problem.Σ[fubar_to_ess, fubar_to_ess] + 1e-6 * I).L
    # cholesky_factor = cholesky(problem.Σ + 1e-6 * I).L

    actual_loglikelihood = prior_only ? x -> 0 : x -> benjamin_loglikelihood(problem, αs, βs, dimensions,fubar_to_ess,diagm(ones(402)), x)
    ESS_model = ESSModel(MvNormal(zeros(problem.dimension + 2), diagm(ones(402))), actual_loglikelihood)
    samples = AbstractMCMC.sample(ESS_model, ESS(), n_samples, progress=true)
    transformed_samples = []
    for i in eachindex(samples)
        grid_sample = zeros(402)
        cholesky_factor = cholesky(kernel_matrix_c(problem.grid, exp(samples[i][end] / 2))[fubar_to_ess, fubar_to_ess] + 1e-6 * I).L
        #cholesky_factor = cholesky(kernel_matrix_c(problem.grid, 2) + 1e-6 * I).L
        grid_sample[1:(end-2)] = cholesky_factor * samples[i][1:(end-2)]
        grid_sample[(end-1)] = samples[i][(end-1)]
        grid_sample[end] = exp(samples[i][end]/2)
        push!(transformed_samples, grid_sample)
    end

    grid_samples = [compute_rjess_to_fubar_permutation(supress_vector(transformed_samples[i][1:(end-1)], αs, βs, dimensions),Int64(sqrt(problem.dimension))) for i in eachindex(samples)]
    kernel_parameter_samples = [samples[i][end] for i in eachindex(samples)]
    posterior_mean = mean(grid_samples[1:100:end])
    
    # Create plots
    
    # 1. Posterior mean plot
    posterior_mean_plot = gridplot(
        problem.grid.alpha_ind_vec, 
        problem.grid.beta_ind_vec,
        problem.grid.grid_values, 
        posterior_mean,
        title="Posterior Mean"
    )

    anim = @animate for i in collect(1:100:length(grid_samples))
        gridplot(
            problem.grid.alpha_ind_vec, 
            problem.grid.beta_ind_vec, 
            problem.grid.grid_values, 
            grid_samples[i]
        )
    end

    kernel_parameter_plot = plot(exp.(kernel_parameter_samples[1:100:end] ./ 2), title = "Kernel scaling factor")
    
    # Add kernel bandwidth titration animation
    # Compute the pre-transformation posterior mean (average of raw samples)
    raw_posterior_mean = mean(samples[1:100:end])
    
    # Create animation showing effect of different kernel bandwidths
    bandwidth_anim = @animate for c in 1.0:0.05:5
        # For each bandwidth value, transform the raw posterior mean
        grid_sample = zeros(402)
        cholesky_factor = cholesky(kernel_matrix_c(problem.grid, c)[fubar_to_ess, fubar_to_ess] + 1e-6 * I).L
        grid_sample[1:(end-2)] = cholesky_factor * raw_posterior_mean[1:(end-2)]
        grid_sample[(end-1)] = raw_posterior_mean[(end-1)]
        
        # Apply suppression and permutation
        transformed_sample = compute_rjess_to_fubar_permutation(
            supress_vector(grid_sample[1:(end-1)], αs, βs, dimensions),
            Int64(sqrt(problem.dimension))
        )
        
        # Plot with current bandwidth value
        gridplot(
            problem.grid.alpha_ind_vec, 
            problem.grid.beta_ind_vec, 
            problem.grid.grid_values, 
            transformed_sample,
            title="Kernel Bandwidth c = $c"
        )
    end
    
    # Generate Cholesky factor plots for specific bandwidth values
    cholesky_plots = []
    for c in [1.0, 4.0, 10.0]
        # Compute kernel matrix and its Cholesky factor
        K = kernel_matrix_c(problem.grid, c)[fubar_to_ess, fubar_to_ess] + 1e-6 * I
        L = cholesky(K).L
        
        # Create a more informative visualization
        # 1. Main heatmap with better color scaling
        chol_heatmap = heatmap(
            Matrix(L), 
            title="Cholesky Factor (c = $c)",
            color=:viridis,
            clim=(0, maximum(L)/4),  # Adjust color scale to highlight structure
            aspect_ratio=:equal,
            framestyle=:box,
            xlabel="Column index",
            ylabel="Row index"
        )
        
        # 2. Add a plot showing the sparsity pattern
        sparsity_pattern = abs.(L) .> 1e-3
        sparsity_plot = spy(Matrix(L), 
            title="Sparsity Pattern (c = $c)",
            markersize=1,
            color=:blues,
            xlabel="Column index",
            ylabel="Row index"
        )
        
        # 3. Add a plot showing the diagonal values
        diag_plot = plot(
            diag(L), 
            title="Diagonal Values (c = $c)",
            xlabel="Index",
            ylabel="Value",
            legend=false,
            linewidth=2
        )
        
        # 4. Add a plot showing a few selected rows
        row_indices = [1, 100, 200, 300, 400]
        row_plot = plot(
            title="Selected Rows of L (c = $c)",
            xlabel="Column index",
            ylabel="Value",
            legend=:topright
        )
        
        for (i, idx) in enumerate(row_indices)
            if idx <= size(L, 1)
                row_values = L[idx, 1:idx]
                plot!(row_plot, 1:idx, row_values, 
                      label="Row $idx", 
                      linewidth=2,
                      marker=:circle,
                      markersize=3)
            end
        end
        
        # Combine the plots in a 2x2 layout
        combined_plot = plot(chol_heatmap, sparsity_plot, diag_plot, row_plot, 
                            layout=(2,2), 
                            size=(800, 800),
                            plot_title="Cholesky Factor Analysis (c = $c)")
        
        push!(cholesky_plots, combined_plot)
    end

    return posterior_mean_plot, kernel_parameter_plot, anim, bandwidth_anim, cholesky_plots
end
