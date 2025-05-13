struct MALAWithinGibbs
    grids :: Vector{FUBARGrid}
    ν0    :: Float64
    S0    :: Matrix{Float64}
    τ     :: Float64
end

# ——— likelihood + gradient for one μ_i block ———

"""
Compute log‐likelihood for a single μ against its grid.
"""
function loglik_i(μ::Vector{Float64}, L::Matrix{Float64})
    log_soft = μ .- logsumexp(μ)
    soft     = exp.(log_soft)
    return sum(log.(soft' * L))
end

"""
Gradient of log‐posterior w.r.t. μ_i:
   ∇ℓ + ∇ log N(μ | 0, Σ)
"""
function grad_logpost_μ(μ::Vector{Float64},
                        L::Matrix{Float64},
                        Σinv::Matrix{Float64})
    K, N = size(L)
    # data‐term
    log_soft = μ .- logsumexp(μ)
    soft     = exp.(log_soft)
    gℓ = zeros(K)
    @inbounds for j in 1:N
        Lj    = view(L, :, j)
        logn  = log_soft .+ log.(Lj)
        denom = logsumexp(logn)
        pj    = exp.(logn .- denom)
        gℓ   .+= pj .- soft * sum(pj)
    end
    # prior term
    return gℓ .- Σinv * μ
end

# ——— one MALA update for μ_i ———

"""
In‐place MALA update for μ given its grid and current Σ.
"""
function mala_update!(μ::Vector{Float64},
                      grid::FUBARGrid,
                      Σ::Matrix{Float64},
                      τ::Float64)
    L      = grid.cond_lik_matrix
    Σinv   = inv(Σ)
    # current gradient + drift
    g      = grad_logpost_μ(μ, L, Σinv)
    drift  = (τ/2) * g
    # propose
    ξ      = randn(length(μ))
    μ_prop = μ .+ drift .+ sqrt(τ) .* ξ

    # compute log‐target
    lp     = loglik_i(μ, L) - 0.5 * dot(μ, Σinv * μ)
    lp_prop= loglik_i(μ_prop, L) - 0.5 * dot(μ_prop, Σinv * μ_prop)

    # forward/reverse proposal log‐densities
    function logq(to, fr, drift)
        m = fr .+ drift
        -0.5*dot(to .- m, to .- m)/τ - length(fr)/2*log(2π*τ)
    end
    logq_fwd = logq(μ_prop, μ, drift)
    # reverse drift at μ_prop
    g_prop   = grad_logpost_μ(μ_prop, L, Σinv)
    drift_r  = (τ/2) * g_prop
    logq_rev = logq(μ, μ_prop, drift_r)

    # MH accept
    logα = (lp_prop + logq_rev) - (lp + logq_fwd)
    if log(rand()) < logα
        μ .= μ_prop
        return true
    else
        return false
    end
end

# ——— conjugate Wishart draw for Σ ———

"""
Given current μs, draw Σ = Λ⁻¹ with
    Λ ~ Wishart(ν0 + n, S0 + ∑ μμᵀ).
"""
function sample_Sigma(μs::Vector{Vector{Float64}},
                      ν0::Float64,
                      S0::Matrix{Float64})
    n = length(μs)
    K = length(μs[1])
    S = zeros(K, K)
    for μ in μs
        S .+= μ * μ'
    end
    df    = ν0 + n
    scale = S0 .+ S
    Λ     = rand(Wishart(df, scale))
    return inv(Λ)
end

# ——— full Gibbs sampler ———

"""
Run MALA-within-Gibbs:

- grids, ν0, S0, τ from the sampler struct
- μ0s: initial μs (Vector of K‐vectors)
- Σ0: initial Σ
- nsamples, burnin
"""
function run_gibbs(s::MALAWithinGibbs,
                   μ0s::Vector{Vector{Float64}},
                   Σ0::Matrix{Float64},
                   nsamples::Int;
                   burnin::Int=0)

    G = length(s.grids)
    # chains
    μ_chain = [Vector{Vector{Float64}}() for _ in 1:G]
    Σ_chain = Matrix{Float64}[]

    # init
    μs = deepcopy(μ0s)
    Σ  = copy(Σ0)

    # burn-in
    for _ in 1:burnin
        for i in 1:G
            mala_update!(μs[i], s.grids[i], Σ, s.τ)
        end
        Σ = sample_Sigma(μs, s.ν0, s.S0)
    end

    # sampling
    for _ in 1:nsamples
        # update each μ_i
        for i in 1:G
            mala_update!(μs[i], s.grids[i], Σ, s.τ)
            push!(μ_chain[i], copy(μs[i]))
        end
        # update Σ
        Σ = sample_Sigma(μs, s.ν0, s.S0)
        push!(Σ_chain, copy(Σ))
    end

    return μ_chain, Σ_chain
end