using Random
using StatsFuns: logsumexp
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers on the SPD manifold P(n)
# ─────────────────────────────────────────────────────────────────────────────
expmap_spd(X, V) = begin
    Xh = sqrt(X)
    invXh = inv(Xh)
    Xh * exp(invXh * V * invXh) * Xh
end

logmap_spd(X, Y) = begin
    Xh = sqrt(X)
    invXh = inv(Xh)
    Xh * log(invXh * Y * invXh) * Xh
end

geo_drift(X) = -(size(X, 1) + 1) / 2 * X

# ─────────────────────────────────────────────────────────────────────────────
#  Problem definition
# ─────────────────────────────────────────────────────────────────────────────
abstract type AbstractHierarchicalRMALAProblem end

struct HierarchicalRMALAWishart <: AbstractHierarchicalRMALAProblem
    grids::Vector{FUBARgrid}
    Σ0::Matrix{Float64}    # Wishart scale
    ν0::Float64            # Wishart degrees of freedom
end

struct HierarchicalRMLALKJ <: AbstractHierarchicalRMALAProblem
    grids::Vector{FUBARgrid}
    η::Float64             # LKJ shape parameter
end

num_grids(p::AbstractHierarchicalRMALAProblem) = length(p.grids)

# ─────────────────────────────────────────────────────────────────────────────
#  Softmax‐likelihood for all μ’s
# ─────────────────────────────────────────────────────────────────────────────
function loglikelihood(p::AbstractHierarchicalRMALAProblem,
    μs::Vector{Vector{Float64}})
    ll = 0.0
    for (i, μ) in enumerate(μs)
        L = p.grids[i].cond_lik_matrix  # K×N
        log_soft = μ .- logsumexp(μ)
        soft = exp.(log_soft)
        ll += sum(log.(soft' * L))
    end
    return ll
end

# ─────────────────────────────────────────────────────────────────────────────
#  Wishart prior on Σ
# ─────────────────────────────────────────────────────────────────────────────
function logprior_Σ(p::HierarchicalRMALAWishart, Σ::Matrix{Float64})
    n, ν0 = size(Σ, 1), p.ν0
    F = cholesky(Symmetric(Σ))
    logdetΣ = 2 * sum(log, diag(F.U))
    return ((ν0 - n - 1) / 2) * logdetΣ - 0.5 * tr(inv(p.Σ0) * Σ)
end

function gradE_logp_Σ(p::HierarchicalRMALAWishart,
    μs::Vector{Vector{Float64}},
    Σ::Matrix{Float64})
    n, ν0 = size(Σ, 1), p.ν0
    Σ0inv = inv(p.Σ0)
    Σinv = inv(Σ)
    S = zeros(n, n)
    for μ in μs
        S .+= μ * μ'
    end
    return ((ν0 - n - 1) / 2) * Σinv .- 0.5 * Σ0inv .+ 0.5 * Σinv * S * Σinv
end

# ─────────────────────────────────────────────────────────────────────────────
#  Extended χ–LKJ prior on Σ
# ─────────────────────────────────────────────────────────────────────────────
function logprior_Σ(p::HierarchicalRMLALKJ, Σ::Matrix{Float64})
    n = size(Σ, 1)
    d = sqrt.(diag(Σ))
    D = Diagonal(d)
    R = inv(D) * Σ * inv(D)
    k = size(p.grids[1].cond_lik_matrix, 1)
    term1 = (k - n - 1) * sum(log.(d)) - 0.5 * sum(d .^ 2)
    valR, signR = logdet(R)
    term2 = (p.η - 1) * valR
    return term1 + term2
end

function gradE_logp_Σ(p::HierarchicalRMLALKJ,
    μs::Vector{Vector{Float64}},
    Σ::Matrix{Float64})
    n = size(Σ, 1)
    d = sqrt.(diag(Σ))
    Dinv2 = Diagonal(1 ./ (d .^ 2))
    k = size(p.grids[1].cond_lik_matrix, 1)
    # diagonal part
    Gdiag = ((k - n - 1) ./ (2d)) .- 0.5
    G1 = Diagonal(Gdiag)
    # correlation part
    G2 = (p.η - 1) * (inv(Σ) - Dinv2)
    return G1 .+ G2
end

# universal Riemannian‐gradient lift
gradR_logp_Σ(p, μs, Σ) = Σ * gradE_logp_Σ(p, μs, Σ) * Σ

# ─────────────────────────────────────────────────────────────────────────────
#  Euclidean gradient wrt each μ_i (softmax likelihood + Gaussian prior)
# ─────────────────────────────────────────────────────────────────────────────
function gradE_logp_μs(p::AbstractHierarchicalRMALAProblem,
    μs::Vector{Vector{Float64}},
    Σ::Matrix{Float64})
    Σinv = inv(Σ)
    G = Vector{Vector{Float64}}(undef, num_grids(p))
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
        G[i] = gℓ .- Σinv * μ
    end
    return G
end

# ─────────────────────────────────────────────────────────────────────────────
#  Full log‐posterior
# ─────────────────────────────────────────────────────────────────────────────
function logposterior(p::AbstractHierarchicalRMALAProblem,
    μs::Vector{Vector{Float64}},
    Σ::Matrix{Float64})
    ℓ = loglikelihood(p, μs)
    lpΣ = logprior_Σ(p, Σ)
    lpμ = sum(-0.5 * dot(μ, inv(Σ) * μ) for μ in μs)
    return ℓ + lpΣ + lpμ
end

# ─────────────────────────────────────────────────────────────────────────────
#  One joint RMALA step
# ─────────────────────────────────────────────────────────────────────────────
function rmala_step(p::AbstractHierarchicalRMALAProblem,
    μs::Vector{Vector{Float64}},
    Σ::Matrix{Float64};
    τμ::Real=1e-2,
    τΣ::Real=1e-2)

    # Euclidean drift for μ, Riemannian drift for Σ
    Gμ = gradE_logp_μs(p, μs, Σ)
    drift_μ = [0.5 * g for g in Gμ]
    drift_Σ = 0.5 * gradR_logp_Σ(p, μs, Σ) + geo_drift(Σ)

    # Propose μ
    G = num_grids(p)
    μs_cand = Vector{Vector{Float64}}(undef, G)
    for i in 1:G
        ξ = randn(length(μs[i]))
        μs_cand[i] = μs[i] .+ τμ .* drift_μ[i] .+ sqrt(τμ) .* ξ
    end

    # Propose Σ on the SPD manifold
    n = size(Σ, 1)
    Z = randn(n, n)
    E = (Z + Z') / 2
    sqrtΣ = sqrt(Σ)
    ξΣ = sqrtΣ * E * sqrtΣ
    V = τΣ .* drift_Σ .+ sqrt(τΣ) .* ξΣ
    Σ_cand = expmap_spd(Σ, V)

    # log‐proposal densities
    logq_euc(to, fr, dr, τ) = begin
        δ = to .- fr .- τ .* dr
        d = length(fr)
        -0.5 * d * log(2π * τ) - dot(δ, δ) / (2τ)
    end
    logq_spd(to, fr, dr, τ) = begin
        V1 = logmap_spd(fr, to)
        S1 = V1 .- τ .* dr
        d = size(fr, 1) * (size(fr, 1) + 1) ÷ 2
        F = cholesky(Symmetric(fr))
        Y = F.U \ (F.L \ S1)
        quad = tr(Y * Y)
        -0.5 * d * log(2π * τ) + 0.5 * (size(fr, 1) + 1) * sum(log, diag(F.U)) - quad / (2τ)
    end

    logq_fwd = sum(logq_euc.(μs_cand, μs, drift_μ, τμ)) +
               logq_spd(Σ_cand, Σ, drift_Σ, τΣ)

    # reverse drifts
    Gμ_p = gradE_logp_μs(p, μs_cand, Σ_cand)
    drift_μ_p = [0.5 * g for g in Gμ_p]
    drift_Σ_p = 0.5 * gradR_logp_Σ(p, μs_cand, Σ_cand) + geo_drift(Σ_cand)

    logq_rev = sum(logq_euc.(μs, μs_cand, drift_μ_p, τμ)) +
               logq_spd(Σ, Σ_cand, drift_Σ_p, τΣ)

    # accept/reject
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
#  RMALA sampler
# ─────────────────────────────────────────────────────────────────────────────
function run_rmala(p::AbstractHierarchicalRMALAProblem,
    Σ0::Matrix{Float64},
    nsamples::Integer;
    μ0s::Union{Nothing,Vector{Vector{Float64}}}=nothing,
    burnin::Integer=1_000,
    τμ::Real=1e-4,
    τΣ::Real=1e-7,
    progress::Bool=false)

    G = num_grids(p)
    if μ0s === nothing
        K = size(p.grids[1].cond_lik_matrix, 1)
        μ0s = [zeros(K) for _ in 1:G]
    end

    μ_chains = [Vector{Vector{Float64}}() for _ in 1:G]
    Σ_chain = Matrix{Float64}[]
    logp_chain = Float64[]

    μs, Σ = deepcopy(μ0s), copy(Σ0)
    accepted = total = 0

    # burn‐in
    for k in 1:burnin
        μs, Σ, ok = rmala_step(p, μs, Σ; τμ=τμ, τΣ=τΣ)
        accepted += ok
        total += 1
        if progress && k % 100 == 0
            println("Burn‐in $k/$burnin – acc rate $(accepted/total)")
        end
    end

    # sampling
    accepted = total = 0
    for k in 1:nsamples
        μs, Σ, ok = rmala_step(p, μs, Σ; τμ=τμ, τΣ=τΣ)
        total += 1
        if ok
            accepted += 1
            for i in 1:G
                push!(μ_chains[i], copy(μs[i]))
            end
            push!(Σ_chain, copy(Σ))
            push!(logp_chain, logposterior(p, μs, Σ))
        end
        if progress && k % 100 == 0
            println("Sampling $k/$nsamples – acc rate $(accepted/total)")
        end
    end

    stats = (accept_rate=accepted / total, τμ=τμ, τΣ=τΣ, burnin=burnin)
    return μ_chains, Σ_chain, logp_chain, stats
end
