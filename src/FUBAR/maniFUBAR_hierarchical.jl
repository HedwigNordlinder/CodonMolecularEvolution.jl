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
#  Problem types
# ─────────────────────────────────────────────────────────────────────────────
abstract type AbstractHierarchicalRMALAProblem end

struct HierarchicalRMALAWishart <: AbstractHierarchicalRMALAProblem
    grids::Vector{FUBARgrid}
    Σ0::Matrix{Float64}    # Wishart scale
    ν0::Float64            # Wishart dof
end

struct HierarchicalRMLALKJ <: AbstractHierarchicalRMALAProblem
    grids::Vector{FUBARgrid}
    η::Float64             # LKJ shape
    k::Int                  # χ dof
end

struct HierarchicalRMALARiemGauss <: AbstractHierarchicalRMALAProblem
    grids::Vector{FUBARgrid}
    σ::Float64     # Riemannian "scale"
    M0::Matrix{Float64}  # Center point on SPD manifold
end

# Default constructor with identity matrix as center
function HierarchicalRMALARiemGauss(grids::Vector{FUBARgrid}, σ::Float64)
    n = length(grids[1].grid)  # Assuming all grids have same dimension
    HierarchicalRMALARiemGauss(grids, σ, Matrix{Float64}(I, n, n))
end

num_grids(p::AbstractHierarchicalRMALAProblem) = length(p.grids)

# ─────────────────────────────────────────────────────────────────────────────
#  Softmax likelihood part (identical)
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
    logdetΣ = 2sum(log, diag(F.U))
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

function logprior_Σ(p::HierarchicalRMALARiemGauss, Σ)
    # Riemannian‐Gaussian part
    T = logmap_spd(p.M0, Σ)
    gauss = -0.5 / p.σ^2 * tr(T * T)

    # normalization from μ|Σ priors:  −½ ∑_i K_i log det(Σ)
    Ktotal = sum(size(g.cond_lik_matrix,1) for g in p.grids)
    norm  = -0.5 * Ktotal * logdet(Σ)

    return gauss + norm
end

function gradE_logp_Σ(
    p::HierarchicalRMALARiemGauss,
    μs::Vector{Vector{Float64}},
    Σ::Matrix{Float64}
)
    # 1) the “Gaussian” Riemann‐prior piece
    σ2   = p.σ^2
    M0h  = sqrt(p.M0)
    iM0h = inv(M0h)
    X    = iM0h * Σ * iM0h
    T    = log(X)
    Gx   = inv(X) * T               # Frechet-adjoint term
    G_gauss = -(1/σ2) * (iM0h * Gx * iM0h)

    # 2) the μ–Σ coupling term
    S = zeros(size(Σ))
    for μ in μs
        S .+= μ * μ'
    end
    G_mu = 0.5 * inv(Σ) * S * inv(Σ)

    # 3) the missing normalization gradient: −½K Σ⁻¹
    Ktotal = sum(length(μ) for μ in μs)
    G_norm = -(Ktotal/2) * inv(Σ)

    return G_gauss .+ G_mu .+ G_norm
end
# ----------------------------------------------------------------------------
#  Extended χ–LKJ prior on Σ = D R D  with d_i∼χ(k),  R∼LKJ(η)
#  p(Σ) ∝ ∏ d_i^(k−n−1) e^(−d_i^2/2) ⋅ (det R)^(η−1)
# ----------------------------------------------------------------------------
function logprior_Σ(p::HierarchicalRMLALKJ, Σ::Matrix{Float64})
    n = size(Σ, 1)
    # extract d_i = sqrt(Σ_ii)
    d = sqrt.(diag(Σ))
    D = Diagonal(d)
    # correlation matrix
    R = inv(D) * Σ * inv(D)
    # dimensionality of each μ grid
    k = p.k

    # ∑[(k−n−1)·log d_i  − ½ d_i^2]
    term1 = sum((k - n - 1) .* log.(d) .- 0.5 .* (d .^ 2))

    # Julia's logdet(R) returns a single Float64 for SPD matrices
    term2 = (p.η - 1) * logdet(R)

    return term1 + term2
end

function gradE_logp_Σ(p::HierarchicalRMLALKJ,
    μs::Vector{Vector{Float64}},
    Σ::Matrix{Float64})
    n = size(Σ, 1)
    # precompute
    Σinv = inv(Σ)
    diagΣ = diag(Σ)
    k = p.k

    # 1) the χ–scale + LKJ parts
    #    ∂/∂Σ_ii of ∑[(k−n−1) log d_i − ½ d_i^2]
    diag_grad = ((k - n - 1) ./ (2 .* diagΣ)) .- 0.5
    G1 = Diagonal(diag_grad)
    #    ∂/∂Σ of (η−1)·log det(R) = (η−1)(Σ⁻¹ − D⁻²)
    G2 = (p.η - 1) .* (Σinv .- Diagonal(1 ./ (diagΣ)))

    # 2) the μ–Σ coupling term: ∑_i ∇_Σ [−½ μᵢᵀ Σ⁻¹ μᵢ] = +½ Σ⁻¹ S Σ⁻¹
    S = zeros(n, n)
    for μ in μs
        S .+= μ * μ'
    end
    Gμ = 0.5 * Σinv * S * Σinv

    return G1 .+ G2 .+ Gμ
end


# lift to Riemannian gradient
gradR_logp_Σ(p, μs, Σ) = Σ * gradE_logp_Σ(p, μs, Σ) * Σ

# ─────────────────────────────────────────────────────────────────────────────
#  Euclidean gradient wrt μ_i (same as before)
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
            logn = log_soft .+ log.(Lj)
            denom = logsumexp(logn)
            pj = exp.(logn .- denom)
            gℓ .+= pj .- soft * sum(pj)
        end
        G[i] = gℓ .- Σinv * μ
    end
    return G
end

# ─────────────────────────────────────────────────────────────────────────────
#  Full log-posterior
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

    # 1) drifts
    Gμ = gradE_logp_μs(p, μs, Σ)
    drift_μ = [0.5 * g for g in Gμ]
    drift_Σ = 0.5 * gradR_logp_Σ(p, μs, Σ) + geo_drift(Σ)

    # 2a) propose μ
    G = num_grids(p)
    μs_cand = Vector{Vector{Float64}}(undef, G)
    for i in 1:G
        ξ = randn(length(μs[i]))
        μs_cand[i] = μs[i] .+ τμ .* drift_μ[i] .+ sqrt(τμ) .* ξ
    end

    # 2b) propose Σ on SPD
    n = size(Σ, 1)
    Z = randn(n, n)
    E = (Z + Z') / 2
    sqrtΣ = sqrt(Σ)
    ξΣ = sqrtΣ * E * sqrtΣ
    V = τΣ .* drift_Σ .+ sqrt(τΣ) .* ξΣ
    Σ_cand = expmap_spd(Σ, V)

    # 3) log-q forward/reverse
    logq_euc(to, fr, dr, τ) = begin
        δ = to .- fr .- τ .* dr
        d = length(fr)
        -0.5 * d * log(2π * τ) - dot(δ, δ) / (2τ)
    end
    logq_spd(to, fr, dr, τ) = begin
        V1 = logmap_spd(fr, to)
        S1 = V1 .- τ .* dr
        n = size(fr, 1)
        d = n * (n + 1) ÷ 2
        F = cholesky(Symmetric(fr))
        Y = F.U \ (F.L \ S1)
        quad = tr(Y * Y)
        -0.5 * d * log(2π * τ) +
        (n + 1) * sum(log, diag(F.U)) -
        quad / (2τ)
    end

    logq_fwd = sum(logq_euc.(μs_cand, μs, drift_μ, τμ)) +
               logq_spd(Σ_cand, Σ, drift_Σ, τΣ)

    # reverse‐drift
    Gμ_p = gradE_logp_μs(p, μs_cand, Σ_cand)
    drift_μ_p = [0.5 * g for g in Gμ_p]
    drift_Σ_p = 0.5 * gradR_logp_Σ(p, μs_cand, Σ_cand) + geo_drift(Σ_cand)

    logq_rev = sum(logq_euc.(μs, μs_cand, drift_μ_p, τμ)) +
               logq_spd(Σ, Σ_cand, drift_Σ_p, τΣ)

    # 4) MH‐step
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

    μ_chain = [Vector{Vector{Float64}}() for _ in 1:G]
    Σ_chain = Matrix{Float64}[]
    logp_chain = Float64[]

    μs, Σ = deepcopy(μ0s), copy(Σ0)
    accepted = total = 0

    # burn‐in
    for k in 1:burnin
        μs, Σ, ok = rmala_step(p, μs, Σ; τμ=τμ, τΣ=τΣ)
        total += 1
        accepted += ok
        if progress && k % 100 == 0
            println("Burn‐in $k/$burnin, acc=$(accepted/total)")
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
                push!(μ_chain[i], copy(μs[i]))
            end
            push!(Σ_chain, copy(Σ))
            push!(logp_chain, logposterior(p, μs, Σ))
        end
        if progress && k % 100 == 0
            println("Sampling $k/$nsamples, acc=$(accepted/total)")
        end
    end

    stats = (accept_rate=accepted / total, τμ=τμ, τΣ=τΣ, burnin=burnin)
    return μ_chain, Σ_chain, logp_chain, stats
end
