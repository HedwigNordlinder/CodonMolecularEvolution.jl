###############################################################################
#  Hierarchical RMALA for multiple FUBARgrids sharing a common Σ              #
###############################################################################
using StatsFuns: logsumexp
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers on the SPD manifold P(n)
# ─────────────────────────────────────────────────────────────────────────────
expmap_spd(X, V) = (Xh = sqrt(X); invXh = inv(Xh); Xh * exp(invXh * V * invXh) * Xh)
logmap_spd(X, Y) = (Xh = sqrt(X); invXh = inv(Xh); Xh * log(invXh * Y * invXh) * Xh)
geo_drift(X) = -(size(X, 1) + 1) / 2 * X

# ─────────────────────────────────────────────────────────────────────────────
#  Problem definition: multiple FUBARgrids with a shared covariance matrix Σ
# ─────────────────────────────────────────────────────────────────────────────
struct HierarchicalRMALAProblem
    grids::Vector{FUBARgrid}    # List of FUBARgrids
    Σ0::Matrix{Float64}         # Prior scale (Σ₀) for Σ
    ν0::Float64                 # Degrees of freedom for Σ prior
end

# number of grids and dimension of Σ
num_grids(p::HierarchicalRMALAProblem) = length(p.grids)
dimΣ(p::HierarchicalRMALAProblem) = size(p.Σ0, 1)

# ─────────────────────────────────────────────────────────────────────────────
#  Joint log-posterior of all data, μs, and Σ
# ─────────────────────────────────────────────────────────────────────────────
function logposterior(p::HierarchicalRMALAProblem,
    μs::Vector{Vector{Float64}},
    Σ::Matrix{Float64})
    # data likelihood
    loglik = 0.0
    for (i, grid) in enumerate(p.grids)
        L = grid.cond_lik_matrix   # K × N
        μ = μs[i]
        log_soft = μ .- logsumexp(μ)
        soft = exp.(log_soft)
        loglik += sum(log.(soft' * L))
    end

    # Wishart prior on Σ: p(Σ) ∝ |Σ|^((ν0−n−1)/2) exp[−½ tr(Σ0⁻¹ Σ)]
    n, ν0 = dimΣ(p), p.ν0
    F = cholesky(Symmetric(Σ))
    logdetΣ = 2 * sum(log, diag(F.U))
    logprior_Σ = ((ν0 - n - 1) / 2) * logdetΣ
    -0.5 * tr(inv(p.Σ0) * Σ)

    # Gaussian priors μ_i ~ N(0, Σ)
    Σinv = inv(Σ)
    logprior_μ = 0.0
    for μ in μs
        logprior_μ += -0.5 * dot(μ, Σinv * μ)
    end

    return loglik + logprior_Σ + logprior_μ
end
# ─────────────────────────────────────────────────────────────────────────────
#  Euclidean gradient wrt each μ_i (softmax likelihood + Gaussian prior)
# ─────────────────────────────────────────────────────────────────────────────
function gradE_logp_μs(p::HierarchicalRMALAProblem, μs::Vector{Vector{Float64}}, Σ::Matrix{Float64})
    Σinv = inv(Σ)
    grads = Vector{Vector{Float64}}(undef, num_grids(p))

    for (i, grid) in enumerate(p.grids)
        L = grid.cond_lik_matrix
        K, N = size(L)
        μ = μs[i]
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

        grads[i] = gℓ .- Σinv * μ
    end

    return grads
end

# ─────────────────────────────────────────────────────────────────────────────
#  Riemannian gradient wrt Σ
# ─────────────────────────────────────────────────────────────────────────────
function gradE_logp_Σ(p::HierarchicalRMALAProblem,
    μs::Vector{Vector{Float64}},
    Σ::Matrix{Float64})
    n, ν0 = dimΣ(p), p.ν0
    Σ0inv = inv(p.Σ0)

    # S = ∑ μ μᵀ
    S = zeros(n, n)
    for μ in μs
        S .+= μ * μ'
    end

    Σinv = inv(Σ)

    # grad = Wishart‐term + μ‐term
    #   = (ν0−n−1)/2·Σ⁻¹ − ½·Σ0⁻¹ + ½·Σ⁻¹ S Σ⁻¹
    return ((ν0 - n - 1) / 2) * Σinv .- 0.5 * Σ0inv
    .+0.5 * Σinv * S * Σinv
end
gradR_logp_Σ(p, μs, Σ) = Σ * gradE_logp_Σ(p, μs, Σ) * Σ
gradR_logp_Σ(p, μs, Σ) = Σ * gradE_logp_Σ(p, μs, Σ) * Σ

# ─────────────────────────────────────────────────────────────────────────────
#  One joint RMALA step for hierarchical model
# ─────────────────────────────────────────────────────────────────────────────
function rmala_step(p::HierarchicalRMALAProblem,
    μs, Σ;
    τμ::Real=1e-2,
    τΣ::Real=1e-2)

    # 1) Euclidean drifts for μ, Riemannian drift for Σ
    Gμ = gradE_logp_μs(p, μs, Σ)           # ∇_μ log π
    drift_μ = [0.5 * g for g in Gμ]             # MALA: ε/2·∇
    drift_Σ = 0.5 * gradR_logp_Σ(p, μs, Σ)      # ½·Riemannian ∇
    +geo_drift(Σ)                   # + geometric penalty

    # 2a) Propose μ (classical MALA)
    μs_cand = Vector{Vector{Float64}}(undef, num_grids(p))
    for i in 1:num_grids(p)
        ξμ = randn(length(μs[i]))
        μs_cand[i] = μs[i] .+ τμ .* drift_μ[i] .+ sqrt(τμ) .* ξμ
    end

    # 2b) Propose Σ on the SPD manifold
    n = dimΣ(p)
    Z = randn(n, n)
    E = (Z + Z') / 2                     # isotropic tangent noise
    sqrtΣ = sqrt(Σ)
    ξΣ = sqrtΣ * E * sqrtΣ
    V = τΣ .* drift_Σ .+ sqrt(τΣ) .* ξΣ
    Σ_cand = expmap_spd(Σ, V)

    # 3) Log proposal densities (variance=ε)
    function logq_euc(to, from, drift, τ)
        δ = to .- from .- τ .* drift
        D = length(from)
        return -0.5 * D * log(2π * τ) - dot(δ, δ) / (2τ)
    end
    function logq_spd(to, from, drift, τ)
        V1 = logmap_spd(from, to)
        S1 = V1 .- τ .* drift
        d = size(from, 1) * (size(from, 1) + 1) ÷ 2
        F = cholesky(Symmetric(from))
        # solve Y = Σ⁻¹·S1 via two triangular solves
        Y = F.U \ (F.L \ S1)
        quad = tr(Y * Y)
        return -0.5 * d * log(2π * τ)
        +0.5 * (size(from, 1) + 1) * sum(log, diag(F.U))
        -quad / (2τ)
    end

    # 4) Accept/reject
    logq_fwd = sum(logq_euc.(μs_cand, μs, drift_μ, τμ)) +
               logq_spd(Σ_cand, Σ, drift_Σ, τΣ)

    Gμ_p = gradE_logp_μs(p, μs_cand, Σ_cand)
    drift_μ_p = [0.5 * g for g in Gμ_p]
    drift_Σ_p = 0.5 * gradR_logp_Σ(p, μs_cand, Σ_cand) + geo_drift(Σ_cand)

    logq_rev = sum(logq_euc.(μs, μs_cand, drift_μ_p, τμ)) +
               logq_spd(Σ, Σ_cand, drift_Σ_p, τΣ)

    logα = logposterior(p, μs_cand, Σ_cand) -
           logposterior(p, μs, Σ) +
           logq_rev - logq_fwd

    if log(rand()) < logα
        return μs_cand, Σ_cand, true
    else
        return μs, Σ, false
    end
end

# ─────────────────────────────────────────────────────────────────────────────
#  RMALA sampler for hierarchical model
# ─────────────────────────────────────────────────────────────────────────────
"""
    run_rmala(p, Σ0, nsamples; μ0s=nothing, burnin=1000, τμ=1e-4, τΣ=1e-7, progress=false)

Samples from the hierarchical RMALA posterior.

If μ0s is not provided, each μ_i is initialized to zero.

Returns:
  μ_chains :: Vector{Vector{Vector{Float64}}} of length G (num_grids).
                Each μ_chains[i] contains only accepted samples for grid i.
  Σ_chain  :: Vector{Matrix{Float64}} containing only accepted Σ samples
  stats     :: Named tuple with fields:
                 • accept_rate :: overall acceptance fraction
                 • τμ, τΣ      :: step sizes used
                 • burnin     :: number of burnin iterations
"""
function run_rmala(p::HierarchicalRMALAProblem,
    Σ0::Matrix{Float64},
    nsamples::Integer;
    μ0s::Union{Nothing,Vector{Vector{Float64}}}=nothing,
    burnin::Integer=1_000,
    τμ::Real=1e-4,
    τΣ::Real=1e-7,
    progress::Bool=false)
    # initialize μs
    G = num_grids(p)
    if μ0s === nothing
        K = size(p.grids[1].cond_lik_matrix, 1)
        μ0s = [zeros(K) for _ in 1:G]
    end

    # allocate chains
    μ_chains = [Vector{Vector{Float64}}() for _ in 1:G]
    Σ_chain = Matrix{Float64}[]

    # initial state
    μs = deepcopy(μ0s)
    Σ = copy(Σ0)
    accepted = total = 0

    # burn-in phase
    for k in 1:burnin
        μs, Σ, ok = rmala_step(p, μs, Σ; τμ=τμ, τΣ=τΣ)
        accepted += ok
        total += 1
        if progress && k % 100 == 0
            println("Burn-in: $k/$burnin iterations, acceptance rate: $(accepted/total)")
        end
    end

    # sampling phase - run exactly nsamples iterations
    for k in 1:nsamples
        μs, Σ, ok = rmala_step(p, μs, Σ; τμ=τμ, τΣ=τΣ)
        accepted += ok
        total += 1

        if ok
            # Only store accepted samples
            for i in 1:G
                push!(μ_chains[i], copy(μs[i]))
            end
            push!(Σ_chain, copy(Σ))
        end

        if progress && k % 100 == 0
            println("Sampling: $k/$nsamples iterations, acceptance rate: $(accepted/total)")
        end
    end

    stats = (accept_rate=accepted / total,
        τμ=τμ,
        τΣ=τΣ,
        burnin=burnin)

    return μ_chains, Σ_chain, stats
end
