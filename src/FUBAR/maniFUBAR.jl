# Here are some targeted optimizations that should provide notable speedups
# without extensive code changes:

#=
1. CACHE CHOLESKY FACTORS OF Σ0 IN THE PROBLEM STRUCT
   - This avoids recomputing it in every iteration
=#

# Add this field to your RMALAProblem struct definition:
struct RMALAProblem
    grid::FUBARgrid
    Σ0::Matrix{Float64}
    ν0::Float64
    Σ0_chol::Cholesky{Float64}  # Add this field
    
    # Add a constructor that pre-computes this value
    function RMALAProblem(grid, Σ0, ν0)
        Σ0_chol = cholesky(Symmetric(Σ0))
        return new(grid, Σ0, ν0, Σ0_chol)
    end
end

#=
2. MODIFY LOGPOSTERIOR TO USE THE CACHED CHOLESKY FACTOR
   - Avoid recomputing the Cholesky factorization of Σ0
=#

function logposterior(p::RMALAProblem, μ, Σ)
    L = p.grid.cond_lik_matrix

    # ----- data likelihood (unchanged) --------------------------------------
    log_soft = μ .- logsumexp(μ)
    soft = exp.(log_soft)
    loglik = sum(log.(soft' * L))

    # ----- priors (use cached Cholesky for Σ0) ------------------------------
    n, ν0 = dim(p), p.ν0
    F = cholesky(Symmetric(Σ))
    logdetΣ = 2 * sum(log, diag(F.U))
    Σinvμ = F \ (F' \ μ)

    # Use cached Cholesky factor for Σ0
    logprior_Σ = -0.5 * (ν0 + n + 1) * logdetΣ -
                 0.5 * sum(abs2, F \ p.Σ0_chol.U)  # Use cached Cholesky

    logprior_μ = -0.5 * dot(μ, Σinvμ)

    return loglik + logprior_Σ + logprior_μ
end

#=
3. REUSE COMPUTED VALUES IN RMALA_STEP
   - Avoid recomputing values that are used multiple times
   - Pass computed matrices between functions
=#

function rmala_step(prob::RMALAProblem, μ, Σ; τμ=1e-2, τΣ=1e-2)
    # Compute Cholesky once and reuse it
    F_Σ = cholesky(Symmetric(Σ))
    
    # Compute square root once and reuse it
    sqrtΣ = sqrt(Symmetric(Σ))
    
    # ─── current gradients and drifts ───────────────────────────────────────
    # Pass the Cholesky factor to avoid recomputation
    gμ = gradE_logp_μ(prob, μ, Σ, F_Σ)
    gΣR = gradR_logp_Σ(prob, μ, Σ, F_Σ)

    drift_μ = 0.5 * gμ
    drift_Σ = 0.5 * gΣR + geo_drift(Σ)

    # ─── propose μᶜ  (Euclidean MALA) ────────────────────────────────────────
    ξμ = randn(length(μ))
    μcand = μ + τμ * drift_μ + sqrt(2τμ) * ξμ

    # ─── propose Σᶜ  (SPD R-MALA) ────────────────────────────────────────────
    n = dim(prob)
    E = randn(n, n)
    E = (E + E') / 2
    ξΣ = sqrtΣ * E * sqrtΣ
    
    V = τΣ * drift_Σ + sqrt(2τΣ) * ξΣ
    Σcand = expmap_spd(Σ, V)
    
    # Compute Cholesky of candidate once and reuse
    F_Σcand = cholesky(Symmetric(Σcand))

    # ─── log proposal densities - pass Cholesky factors ─────────────────────
    # Compute log posterior with cached Cholesky factors
    logp_cur = logposterior(prob, μ, Σ, F_Σ)
    logp_cand = logposterior(prob, μcand, Σcand, F_Σcand)
    
    # Euclidean part
    logq_cur2prop = logq_euc(μcand, μ, drift_μ, τμ) +
                    logq_spd(Σcand, Σ, drift_Σ, τΣ, F_Σ)  # Pass F_Σ

    # recompute drifts at the proposal, reusing Cholesky
    drift_μp = 0.5 * gradE_logp_μ(prob, μcand, Σcand, F_Σcand)
    drift_Σp = 0.5 * gradR_logp_Σ(prob, μcand, Σcand, F_Σcand) + geo_drift(Σcand)

    logq_prop2cur = logq_euc(μ, μcand, drift_μp, τμ) +
                    logq_spd(Σ, Σcand, drift_Σp, τΣ, F_Σcand)  # Pass F_Σcand

    # ─── acceptance ratio ───────────────────────────────────────────────────
    logα = logp_cand - logp_cur + logq_prop2cur - logq_cur2prop

    if log(rand()) < logα
        return μcand, Σcand, true
    else
        return μ, Σ, false
    end
end

#=
4. MODIFY HELPER FUNCTIONS TO ACCEPT PRECOMPUTED VALUES
=#

# Add optional Cholesky parameter to logposterior
function logposterior(p::RMALAProblem, μ, Σ, F=nothing)
    L = p.grid.cond_lik_matrix

    # ----- data likelihood ---------------------------------------------------
    log_soft = μ .- logsumexp(μ)
    soft = exp.(log_soft)
    loglik = sum(log.(soft' * L))

    # ----- priors (use cached Cholesky) -------------------------------------
    n, ν0 = dim(p), p.ν0
    
    # Use provided Cholesky factor if available
    if F === nothing
        F = cholesky(Symmetric(Σ))
    end
    
    logdetΣ = 2 * sum(log, diag(F.U))
    Σinvμ = F \ (F' \ μ)

    logprior_Σ = -0.5 * (ν0 + n + 1) * logdetΣ -
                 0.5 * sum(abs2, F \ p.Σ0_chol.U)

    logprior_μ = -0.5 * dot(μ, Σinvμ)

    return loglik + logprior_Σ + logprior_μ
end

# Modify gradE_logp_μ to accept precomputed Cholesky
function gradE_logp_μ(p::RMALAProblem, μ, Σ, F=nothing)
    L = p.grid.cond_lik_matrix
    K, N = size(L)
    log_soft = μ .- logsumexp(μ)
    soft = exp.(log_soft)

    gℓ = zeros(K)
    for j in 1:N
        Lj = @view L[:, j]
        lognum = log_soft .+ log.(Lj)
        denom = logsumexp(lognum)
        pj = exp.(lognum .- denom)
        gℓ .+= pj .- soft * sum(pj)
    end
    
    # Use provided Cholesky if available
    if F === nothing
        F = cholesky(Symmetric(Σ))
    end
    
    gprior = -(F \ (F' \ μ))
    return gℓ + gprior
end

# Modify gradE_logp_Σ to accept precomputed Cholesky
function gradE_logp_Σ(p::RMALAProblem, μ, Σ, F=nothing)
    n, ν0 = dim(p), p.ν0
    
    # Use provided Cholesky if available
    if F === nothing
        F = cholesky(Symmetric(Σ))
    end
    
    Σinv = Matrix(F \ (F' \ I))
    return 0.5 * (ν0 - n - 1) * Σinv - 0.5 * (μ * μ') - 0.5 * Σ0inv(p.Σ0)
end

# Modify gradR_logp_Σ to accept precomputed Cholesky
function gradR_logp_Σ(p, μ, Σ, F=nothing)
    if F === nothing
        return Σ * gradE_logp_Σ(p, μ, Σ) * Σ
    else
        return Σ * gradE_logp_Σ(p, μ, Σ, F) * Σ
    end
end

# Add optional Cholesky parameter to logq_spd
function logq_spd(to, from, drift_from, τ, F=nothing)
    V = logmap_spd(from, to)
    S = V - τ * drift_from
    d = size(from, 1) * (size(from, 1) + 1) ÷ 2
    
    # Use provided Cholesky if available
    if F === nothing
        F = cholesky(Symmetric(from))
    end
    
    Y = F \ (F' \ S)
    quad = tr(Y * Y)
    
    return -0.5 * d * log(4π * τ) + 0.5 * (size(from, 1) + 1) * sum(log, diag(F.U)) - quad / (4τ)
end

#=
5. ADD THINNING TO RUN_RMALA TO REDUCE MEMORY USAGE FOR LARGE MATRICES
=#

function run_rmala(prob::RMALAProblem,
    μ0::AbstractVector,
    Σ0::AbstractMatrix,
    nsamples::Integer;
    burnin::Integer=1_000,
    τμ::Real=1e-4,
    τΣ::Real=1e-7,
    thinning::Integer=1,  # Add thinning parameter
    progress::Bool=false)
    
    dimμ = length(μ0)
    actual_samples = nsamples

    μ_chain = Matrix{Float64}(undef, dimμ, actual_samples)
    Σ_chain = Matrix{Float64}[]

    μ, Σ = copy(μ0), copy(Σ0)
    accepted = 0
    total = 0

    total_iter = burnin + nsamples * thinning
    showprog = progress ? Progress(total_iter; desc="RMALA") : nothing

    sample_idx = 0
    for k in 1:total_iter
        μ, Σ, ok = rmala_step(prob, μ, Σ; τμ=τμ, τΣ=τΣ)
        accepted += ok
        total += 1
        
        # Store samples according to thinning parameter
        if k > burnin && (k - burnin) % thinning == 0
            sample_idx += 1
            μ_chain[:, sample_idx] = μ
            push!(Σ_chain, copy(Σ))
        end
        
        progress && next!(showprog)
    end

    stats = (accept_rate=accepted / total,
        τμ=τμ,
        τΣ=τΣ,
        burnin=burnin,
        thinning=thinning)  # Add thinning to stats

    return μ_chain, Σ_chain, stats
end