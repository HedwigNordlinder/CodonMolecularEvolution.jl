struct RMALAProblem
    grid::FUBARgrid
    Σ0::Matrix{Float64}
    ν0::Float64
    Σ0_chol::Cholesky{Float64}
    Σ0inv_cached::Matrix{Float64}
    
    function RMALAProblem(grid, Σ0, ν0)
        Σ0_chol = cholesky(Symmetric(Σ0))
        Σ0inv_cached = inv(Symmetric(Σ0))
        return new(grid, Σ0, ν0, Σ0_chol, Σ0inv_cached)
    end
end

function rmala_step_optimized(prob::RMALAProblem, μ, Σ; τμ=1e-2, τΣ=1e-2, 
                            work_arrays=nothing)
    n = dim(prob)
    
    # Use pre-allocated work arrays if provided (helpful for 400x400 matrices)
    Σ_temp = work_arrays !== nothing ? work_arrays.Σ_temp : similar(Σ)
    μ_temp = work_arrays !== nothing ? work_arrays.μ_temp : similar(μ)
    
    # ─── Pre-compute expensive decompositions only once ─────────────────────
    # For 400x400 matrices, these are major bottlenecks
    F_Σ = cholesky(Symmetric(Σ))
    
    # Computing the matrix square root is extremely expensive for 400x400
    # Use a more efficient approach for large matrices - compute it once and reuse
    if work_arrays !== nothing && :sqrtΣ in fieldnames(typeof(work_arrays))
        sqrtΣ = work_arrays.sqrtΣ
        # Update the cached square root (only when needed)
        if work_arrays.needs_sqrt_update
            # This is an extremely expensive operation for 400x400
            copyto!(sqrtΣ, sqrt(Symmetric(Σ)))
            work_arrays.needs_sqrt_update = false
        end
    else
        sqrtΣ = sqrt(Symmetric(Σ))  # Very expensive for 400x400
    end
    
    # ─── current gradients and drifts - reuse F_Σ to avoid recalculation ────
    # For 400x400, reducing these calculations has huge impact
    gμ = gradE_logp_μ(prob, μ, Σ, F_Σ)
    
    # Avoid allocating the full gradient matrix if possible
    gΣE = gradE_logp_Σ(prob, μ, Σ, F_Σ)
    
    # Compute Riemannian gradient without creating extra temporaries
    # For 400x400, this matrix multiplication chain is very expensive
    mul!(Σ_temp, Σ, gΣE)  # Σ_temp = Σ * gΣE
    mul!(gΣE, Σ_temp, Σ)  # Reuse gΣE to store the result Σ * gΣE * Σ
    
    drift_μ = 0.5 .* gμ  # Could use in-place scaling for μ_temp if needed
    
    # Compute drift_Σ - reuse Σ_temp to avoid additional allocation
    copyto!(Σ_temp, gΣE)
    rmul!(Σ_temp, 0.5)
    geo = geo_drift(Σ)
    drift_Σ = Σ_temp .+ geo  # This allocates, could be made in-place
    
    # ─── propose μᶜ ────────────────────────────────────────────────────────
    # Generate candidate directly, minimizing intermediate allocations
    for i in eachindex(μ)
        μ_temp[i] = μ[i] + τμ * drift_μ[i] + sqrt(2τμ) * randn()
    end
    μcand = μ_temp  # No copy, just reference
    
    # ─── propose Σᶜ ────────────────────────────────────────────────────────
    # Generate random symmetric matrix - this is unavoidable but not too costly
    E = randn(n, n)
    E = (E + E') / 2
    
    # For 400x400, this is extremely expensive - use pre-allocated memory if possible
    if work_arrays !== nothing && :ξΣ in fieldnames(typeof(work_arrays))
        ξΣ = work_arrays.ξΣ
        mul!(ξΣ, sqrtΣ, E)       # ξΣ = sqrtΣ * E
        mul!(Σ_temp, ξΣ, sqrtΣ)  # Σ_temp = ξΣ * sqrtΣ
        ξΣ = Σ_temp              # Point to result
    else
        ξΣ = sqrtΣ * E * sqrtΣ   # This creates two intermediate matrices - expensive!
    end
    
    # This section is critical for 400x400 matrices - very expensive operations
    # We don't use in-place exponential map because it's complicated and error-prone
    # But we should consider implementing an optimized version for production
    V = τΣ * drift_Σ + sqrt(2τΣ) * ξΣ   # This allocates - unavoidable for now
    Σcand = expmap_spd(Σ, V)            # This is extremely expensive
    F_Σcand = cholesky(Symmetric(Σcand))
    
    # ─── log posterior evaluations - reuse Cholesky factors ─────────────────
    logp_cur = logposterior(prob, μ, Σ, F_Σ)
    logp_cand = logposterior(prob, μcand, Σcand, F_Σcand)
    
    # ─── Forward proposal ───────────────────────────────────────────────────
    # Computing these efficiently is crucial for 400x400 matrices
    logq_cur2prop_euc = logq_euc(μcand, μ, drift_μ, τμ)
    logq_cur2prop_spd = logq_spd(Σcand, Σ, drift_Σ, τΣ, F_Σ)
    logq_cur2prop = logq_cur2prop_euc + logq_cur2prop_spd
    
    # ─── Compute reverse drifts - with minimal recalculation ────────────────
    drift_μp = 0.5 * gradE_logp_μ(prob, μcand, Σcand, F_Σcand)
    
    # Recompute gradient for candidate point (unavoidable)
    gΣE_cand = gradE_logp_Σ(prob, μcand, Σcand, F_Σcand)
    
    # Compute Riemannian gradient without creating extra temporaries
    if work_arrays !== nothing && :Σ_temp2 in fieldnames(typeof(work_arrays))
        temp = work_arrays.Σ_temp2
        mul!(temp, Σcand, gΣE_cand)        # temp = Σcand * gΣE_cand
        mul!(gΣE_cand, temp, Σcand)        # Reuse to store final result
    else
        gΣE_cand = Σcand * gΣE_cand * Σcand  # This allocates three matrices!
    end
    
    # Compute drift - reuse memory for 400x400 case
    drift_Σp = 0.5 * gΣE_cand + geo_drift(Σcand)
    
    # ─── Reverse proposal ───────────────────────────────────────────────────
    logq_prop2cur_euc = logq_euc(μ, μcand, drift_μp, τμ)
    logq_prop2cur_spd = logq_spd(Σ, Σcand, drift_Σp, τΣ, F_Σcand)
    logq_prop2cur = logq_prop2cur_euc + logq_prop2cur_spd
    
    # ─── acceptance ratio ───────────────────────────────────────────────────
    logα = logp_cand - logp_cur + logq_prop2cur - logq_cur2prop
    
    # Update work arrays if we've accepted the proposal
    if log(rand()) < logα
        if work_arrays !== nothing
            work_arrays.needs_sqrt_update = true
        end
        return μcand, Σcand, true
    else
        return μ, Σ, false
    end
end

# Create a struct to hold pre-allocated work arrays for 400x400 case
"""
    RMALAWorkArrays(n, K)

Pre-allocate work arrays for RMALA with matrix dimension n and vector dimension K.
Using these work arrays significantly reduces allocation overhead for large matrices.
"""
struct RMALAWorkArrays
    Σ_temp::Matrix{Float64}    # Temporary matrix same size as Σ
    Σ_temp2::Matrix{Float64}   # Second temporary matrix
    μ_temp::Vector{Float64}    # Temporary vector same size as μ
    sqrtΣ::Matrix{Float64}     # Cache for expensive sqrt(Σ) calculation
    ξΣ::Matrix{Float64}        # For noise term
    needs_sqrt_update::Bool    # Flag to indicate if sqrtΣ needs updating
    
    function RMALAWorkArrays(n::Integer, K::Integer)
        return new(
            zeros(n, n),
            zeros(n, n),
            zeros(K),
            zeros(n, n),
            zeros(n, n),
            true
        )
    end
end

# Optimized helper functions to utilize caching

"""
    logposterior(p, μ, Σ, F=nothing)

Un-normalised joint log-posterior with optional pre-computed Cholesky factor.
Optimized for large (400x400) matrices.
"""
function logposterior(p::RMALAProblem, μ, Σ, F=nothing)
    L = p.grid.cond_lik_matrix
    K, N = size(L)
    
    # ----- data likelihood ---------------------------------------------------
    # For large problems, computing softmax can be done more efficiently
    max_μ = maximum(μ)
    exp_μ = exp.(μ .- max_μ)
    sum_exp_μ = sum(exp_μ)
    soft = exp_μ ./ sum_exp_μ
    
    # Compute log-likelihood more efficiently for large matrices
    # This avoids creating a full K×N matrix for soft'*L
    loglik = 0.0
    @inbounds for j in 1:N
        prob_j = 0.0
        @inbounds for i in 1:K
            prob_j += soft[i] * L[i, j]
        end
        loglik += log(prob_j)
    end
    
    # ----- priors -----------------------------------------------------------
    n, ν0 = dim(p), p.ν0
    
    # Use cached Cholesky if provided - critical for 400x400 matrices
    if F === nothing
        F = cholesky(Symmetric(Σ))
    end
    
    # Calculate log determinant from Cholesky factor - fast
    logdetΣ = 2 * sum(log, diag(F.U))
    
    # Calculate precision * μ efficiently - major bottleneck for 400x400
    # Note: direct triangular solves are faster than F\(F'\μ) in practice
    temp = F.U' \ μ      # temporary vector
    Σinvμ = F.U \ temp   # Σ⁻¹μ
    
    # Calculate trace term efficiently
    # For 400x400, we should cache the Cholesky of Σ0 in the problem struct
    if !hasfield(typeof(p), :Σ0_chol) || p.Σ0_chol === nothing
        Σ0_chol = cholesky(Symmetric(p.Σ0))
    else
        Σ0_chol = p.Σ0_chol
    end
    
    # Compute Σ0⁻¹Σ trace term efficiently (critical for 400x400)
    # This is a major bottleneck in the original code
    # We avoid computing the full inverse and minimize temporaries
    trace_term = 0.0
    U0 = Σ0_chol.U
    @inbounds for j in 1:n
        Uj = F \ @view U0[:,j]  # Solve for each column
        trace_term += sum(abs2, Uj)
    end
    
    logprior_Σ = -0.5 * (ν0 + n + 1) * logdetΣ - 0.5 * trace_term
    logprior_μ = -0.5 * dot(μ, Σinvμ)
    
    return loglik + logprior_Σ + logprior_μ
end

"""
    gradE_logp_Σ(p, μ, Σ, F=nothing)

Euclidean gradient of log-posterior with respect to Σ.
Accepts optional pre-computed Cholesky factor.
Optimized for 400x400 matrices.
"""
function gradE_logp_Σ(p::RMALAProblem, μ, Σ, F=nothing)
    n, ν0 = dim(p), p.ν0
    
    # For 400x400 matrices, we want to avoid full matrix inversion
    # Use cached Cholesky if provided - critical for performance
    if F === nothing
        F = cholesky(Symmetric(Σ))
    end
    
    # For large matrices, computing inverse is extremely expensive
    # Instead of computing the full inverse, we'll compute the needed parts directly
    
    # For (ν0 - n - 1)/2 * Σ⁻¹ term, we construct it column by column
    # instead of forming the full inverse matrix
    grad = similar(Σ)
    
    # First factor: (ν0 - n - 1)/2 * Σ⁻¹
    # We'll build this by computing individual columns
    for j in 1:n
        e_j = zeros(n)
        e_j[j] = 1.0
        # Compute j-th column of inverse
        inv_col = F \ (F' \ e_j)
        # Scale and store
        for i in 1:n
            grad[i,j] = 0.5 * (ν0 - n - 1) * inv_col[i]
        end
    end
    
    # Second factor: -0.5 * μμᵀ 
    # Compute outer product efficiently without temporary
    for j in 1:n
        for i in 1:n
            grad[i,j] -= 0.5 * μ[i] * μ[j]
        end
    end
    
    # Third factor: -0.5 * Σ₀⁻¹
    # For large matrices, we should cache Σ₀⁻¹ in the problem object
    # rather than recomputing it each time
    if !hasfield(typeof(p), :Σ0inv_cached) || p.Σ0inv_cached === nothing
        Σ0inv_val = Σ0inv(p.Σ0)
    else
        Σ0inv_val = p.Σ0inv_cached
    end
    
    # Subtract into the result
    for j in 1:n
        for i in 1:n
            grad[i,j] -= 0.5 * Σ0inv_val[i,j]
        end
    end
    
    return grad
end

"""
    gradE_logp_μ(p, μ, Σ, F=nothing)

Euclidean gradient of log-posterior with respect to μ.
Accepts optional pre-computed Cholesky factor.
Optimized for large vectors and matrices.
"""
function gradE_logp_μ(p::RMALAProblem, μ, Σ, F=nothing)
    L = p.grid.cond_lik_matrix
    K, N = size(L)
    
    # More efficient softmax computation for large vectors
    max_μ = maximum(μ)
    exp_μ = exp.(μ .- max_μ)
    sum_exp_μ = sum(exp_μ)
    soft = exp_μ ./ sum_exp_μ
    
    # Pre-allocate gradient vector - this is an unavoidable allocation
    gℓ = zeros(K)
    
    # For large K,N values, avoid temporary array allocations
    # Each loop iteration should avoid creating new arrays
    for j in 1:N
        # Compute pj more efficiently to avoid temporary allocations
        # This is critical for large K values
        
        # Compute weighted sum and avoid creating arrays
        numer_sum = 0.0
        numer = zeros(K)
        
        @inbounds for i in 1:K
            numer[i] = soft[i] * L[i,j]
            numer_sum += numer[i]
        end
        
        # Update gradient efficiently
        @inbounds for i in 1:K
            pj_i = numer[i] / numer_sum
            gℓ[i] += pj_i - soft[i] * sum(pj_i)
        end
    end
    
    # Use cached Cholesky if provided - critical for 400x400
    if F === nothing
        F = cholesky(Symmetric(Σ))
    end
    
    # Compute prior gradient efficiently
    # Direct triangular solves are more efficient than F\(F'\μ)
    temp = F.U' \ μ
    gprior = -(F.U \ temp)
    
    return gℓ .+ gprior
end

# Optimized run_rmala function
"""
    run_rmala_optimized(prob, μ0, Σ0, nsamples; kwargs...)

Run RMALA with optimizations for large matrices (400x400).
Uses pre-allocated work arrays to minimize allocations during sampling.
"""
function run_rmala_optimized(
    prob::RMALAProblem,
    μ0::AbstractVector,
    Σ0::AbstractMatrix,
    nsamples::Integer;
    burnin::Integer=1_000,
    τμ::Real=1e-4,
    τΣ::Real=1e-7,
    progress::Bool=false,
    thinning::Integer=1
)
    dimμ = length(μ0)
    n = size(Σ0, 1)
    
    # For 400x400 matrices, using pre-allocated work arrays is critical
    work_arrays = RMALAWorkArrays(n, dimμ)
    
    # Calculate actual number of samples to generate
    total_iter = burnin + nsamples * thinning
    actual_samples = nsamples
    
    # Pre-allocate storage - for large problems, this is significant
    μ_chain = Matrix{Float64}(undef, dimμ, actual_samples)
    
    # For 400x400 matrices, storing each Σ matrix requires 1.28 MB
    # Consider using a more memory-efficient storage approach
    # Options:
    # 1. Store only diagonal and upper/lower triangle
    # 2. Store only every k-th sample 
    # 3. Store a low-rank approximation
    
    # Standard approach (uses more memory)
    Σ_chain = Vector{Matrix{Float64}}(undef, actual_samples)
    
    # Initialize
    μ, Σ = copy(μ0), copy(Σ0)
    accepted = 0
    total = 0
    
    # Pre-compute and cache expensive components
    # For 400x400 matrices, caching the Cholesky of Σ0 is a big win
    # Add this to the problem struct if possible, otherwise compute here
    if !hasfield(typeof(prob), :Σ0_chol)
        # Create a temporary cached version
        Σ0_chol = cholesky(Symmetric(prob.Σ0))
        Σ0inv_cached = inv(Symmetric(prob.Σ0))  # Cache this calculation
    end
    
    showprog = progress ? Progress(total_iter; desc="RMALA") : nothing
    
    # Main MCMC loop - this is where most time is spent
    sample_idx = 0
    for k in 1:total_iter
        # Use the optimized step function with work arrays
        μ, Σ, ok = rmala_step_optimized(prob, μ, Σ; τμ=τμ, τΣ=τΣ, work_arrays=work_arrays)
        
        accepted += ok
        total += 1
        
        # Store sample after burnin and according to thinning
        if k > burnin && (k - burnin) % thinning == 0
            sample_idx += 1
            μ_chain[:, sample_idx] = μ
            Σ_chain[sample_idx] = copy(Σ)
        end
        
        progress && next!(showprog)
    end
    
    stats = (
        accept_rate=accepted / total,
        τμ=τμ,
        τΣ=τΣ,
        burnin=burnin,
        thinning=thinning
    )
    
    return μ_chain, Σ_chain, stats
end

# Enhanced problem struct with cached components

# Matrix square root and exponential/logarithm are major bottlenecks
# For 400x400 matrices, consider these optimized variants:

"""
    sqrt_symmetric_optimized(X)

Optimized matrix square root for large symmetric positive definite matrices.
Uses eigendecomposition which is typically faster than the generic algorithm
for 400x400 matrices.
"""
function sqrt_symmetric_optimized(X)
    # For 400x400, eigendecomposition is typically faster than standard sqrt
    d, V = eigen(Symmetric(X))
    # Ensure positive eigenvalues (should be the case for SPD matrices)
    d_sqrt = sqrt.(max.(d, 0))
    return V * Diagonal(d_sqrt) * V'
end

"""
    expmap_spd_optimized(X, V)

Optimized version of the exponential map for large SPD matrices.
Uses cached square root when available.
"""
function expmap_spd_optimized(X, V, sqrtX=nothing)
    if sqrtX === nothing
        sqrtX = sqrt_symmetric_optimized(X)
    end
    invXh = inv(sqrtX)  # This is expensive but unavoidable
    return sqrtX * exp(Symmetric(invXh * V * invXh)) * sqrtX
end

"""
    logmap_spd_optimized(X, Y, sqrtX=nothing)

Optimized version of the logarithmic map for large SPD matrices.
Uses cached square root when available.
"""
function logmap_spd_optimized(X, Y, sqrtX=nothing)
    if sqrtX === nothing
        sqrtX = sqrt_symmetric_optimized(X)
    end
    invXh = inv(sqrtX)  # This is expensive but unavoidable
    return sqrtX * log(Symmetric(invXh * Y * invXh)) * sqrtX
end

# Optimized log proposal density functions
function logq_euc(to, from, drift_from, τ)
    # For 400-length vectors, reducing allocations is important
    # Compute the distance without creating temporary arrays
    k = length(from)
    quad_sum = 0.0
    @inbounds for i in eachindex(to)
        δi = to[i] - from[i] - τ * drift_from[i]
        quad_sum += δi * δi
    end
    return -0.5 * k * log(4π * τ) - quad_sum / (4τ)
end

function logq_spd(to, from, drift_from, τ, F=nothing)
    # This function is a bottleneck for 400x400 matrices
    
    # The matrix logarithm is extremely expensive - unavoidable
    V = logmap_spd(from, to)
    
    # We can avoid additional allocation here
    # For very large matrices, consider using LinearMaps.jl for S = V - τ*drift
    S = V .- τ .* drift_from  # This allocates but is hard to avoid
    
    d = size(from, 1) * (size(from, 1) + 1) ÷ 2
    
    # Use cached Cholesky if provided - critical for 400x400
    if F === nothing
        F = cholesky(Symmetric(from))
    end
    
    # This is a major bottleneck in RMALA - matrix solves are expensive
    # For 400x400, we need to be careful with temporaries
    Y = F \ (F' \ S)  # This creates intermediate arrays we can't easily avoid
    
    # Compute trace more efficiently for large matrices
    quad = 0.0
    @inbounds for i in 1:size(Y, 1)
        @inbounds for j in 1:size(Y, 2)
            quad += Y[i,j]^2
        end
    end
    
    return -0.5 * d * log(4π * τ) + 0.5 * (size(from, 1) + 1) * sum(log, diag(F.U)) - quad / (4τ)
end

# Optimized run_rmala function
function run_rmala_optimized(
    prob::RMALAProblem,
    μ0::AbstractVector,
    Σ0::AbstractMatrix,
    nsamples::Integer;
    burnin::Integer=1_000,
    τμ::Real=1e-4,
    τΣ::Real=1e-7,
    progress::Bool=false,
    thinning::Integer=1
)
    dimμ = length(μ0)
    n = size(Σ0, 1)
    
    # Calculate actual number of samples to generate
    total_iter = burnin + nsamples * thinning
    actual_samples = nsamples
    
    # Pre-allocate storage
    μ_chain = Matrix{Float64}(undef, dimμ, actual_samples)
    Σ_chain = Vector{Matrix{Float64}}(undef, actual_samples)
    
    # Initialize
    μ, Σ = copy(μ0), copy(Σ0)
    accepted = 0
    total = 0
    
    showprog = progress ? Progress(total_iter; desc="RMALA") : nothing
    
    # Main MCMC loop
    sample_idx = 0
    for k in 1:total_iter
        μ, Σ, ok = rmala_step_optimized(prob, μ, Σ; τμ=τμ, τΣ=τΣ)
        accepted += ok
        total += 1
        
        # Store sample after burnin and according to thinning
        if k > burnin && (k - burnin) % thinning == 0
            sample_idx += 1
            μ_chain[:, sample_idx] = μ
            Σ_chain[sample_idx] = copy(Σ)
        end
        
        progress && next!(showprog)
    end
    
    stats = (
        accept_rate=accepted / total,
        τμ=τμ,
        τΣ=τΣ,
        burnin=burnin,
        thinning=thinning
    )
    
    return μ_chain, Σ_chain, stats
end