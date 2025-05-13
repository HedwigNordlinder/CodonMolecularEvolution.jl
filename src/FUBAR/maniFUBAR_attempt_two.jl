# ----------------------------------------------------------------------------
# — abstract "Prior" interface + dispatchers
# ----------------------------------------------------------------------------
abstract type Prior end

# — Forward declaration of the problem type
struct HierarchicalRMALAProblem{Pμ<:Prior, PΣ<:Prior}
    grids::Vector{FUBARgrid}
    prior_mu::Pμ
    prior_sigma::PΣ
end

# — Normal prior on each μᵢ ~ 𝒩(0, Σ)
struct NormalPrior <: Prior end

function logprior(::NormalPrior, μs::Vector{Vector{T}}, Σ::AbstractMatrix{T}) where T
    N, K = length(μs), length(μs[1])
    lp = -N*(K/2*log(2π) + 1/2*logdet(Σ))
    for μ in μs
        lp -= 1/2 * dot(μ, Σ \ μ)
    end
    return lp
end

function grad_logprior(::NormalPrior, μs::Vector{Vector{T}}, Σ::AbstractMatrix{T}) where T
    return [ -(Σ \ μ) for μ in μs ]
end

function euclid_grad_logprior(::NormalPrior, μs::Vector{Vector{T}}, Σ::AbstractMatrix{T}) where T
    N = length(μs)
    S = zeros(eltype(Σ), size(Σ))
    for μ in μs
        S .+= μ * μ'
    end
    t1 = -N/2 * inv(Σ)
    t2 = +1/2 * inv(Σ) * S * inv(Σ)
    return t1 .+ t2
end

# — Wishart prior Σ ~ W(df, V0)
struct WishartPrior{T} <: Prior
    df::Int
    V0::Matrix{T}
end

function logprior(p::WishartPrior, Σ::AbstractMatrix{T}) where T
    ν, V0 = p.df, p.V0; K = size(Σ,1)
    mvlg = (K*(K-1)/4)*log(pi) + sum(lgamma(ν/2 .+ (1 .- collect(1:K))./2))
    return ((ν - K - 1)/2)*logdet(Σ) - 1/2*tr(inv(V0)*Σ) - (ν*K/2)*log(2) - mvlg - (ν/2)*logdet(V0)
end

function grad_logprior(p::WishartPrior, Σ::AbstractMatrix{T}) where T
    ν, V0 = p.df, p.V0; K = size(Σ,1)
    return ((ν - K - 1)/2)*inv(Σ) .- 1/2*inv(V0)
end

# — Riemannian‐Gaussian prior on SPD: p(Σ) ∝ exp(−d_R²(Σ,M)/(2σ²))
struct RiemannianGaussianPrior{T} <: Prior
    M::Matrix{T}
    σ2::T
end

function logprior(p::RiemannianGaussianPrior, Σ::AbstractMatrix{T}) where T
    S  = sqrt(p.M); iS = inv(S)
    Δ  = log(iS * Σ * iS)
    d2 = tr(Δ * Δ)
    return -d2/(2*p.σ2)
end

function grad_logprior(p::RiemannianGaussianPrior, Σ::AbstractMatrix{T}) where T
    S  = sqrt(p.M); iS = inv(S)
    Δ  = log(iS * Σ * iS)
    return -1/p.σ2 * (iS * Δ * iS)
end

# ----------------------------------------------------------------------------
# — likelihood
# ----------------------------------------------------------------------------
function loglikelihood(p::HierarchicalRMALAProblem, μs::Vector{Vector})
    ll = 0.0
    for (i, μ) in enumerate(μs)
        L = p.grids[i].cond_lik_matrix
        log_soft = μ .- logsumexp(μ)
        soft     = exp.(log_soft)
        ll      += sum(log.(soft' * L))
    end
    return ll
end

# ----------------------------------------------------------------------------
# — full problem type
# ----------------------------------------------------------------------------
# Delete this duplicate struct definition
# struct HierarchicalRMALAProblem{Pμ<:Prior, PΣ<:Prior}
#     grids::Vector{FUBARgrid}
#     prior_mu::Pμ
#     prior_sigma::PΣ
# end

# ----------------------------------------------------------------------------
# — posterior + grads
# ----------------------------------------------------------------------------
function log_posterior(p::HierarchicalRMALAProblem, μs, Σ)
    return loglikelihood(p, μs) +
           logprior(p.prior_mu, μs, Σ) +
           logprior(p.prior_sigma, Σ)
end

function grad_log_post_mu(p::HierarchicalRMALAProblem, μs, Σ)
    gs = [ zeros(length(μ)) for μ in μs ]
    for (i, μ) in enumerate(μs)
        L = p.grids[i].cond_lik_matrix
        v = exp.(μ .- logsumexp(μ))
        M = v' * L; w = 1.0 ./ M; gv = L * w; α = dot(gv, v)
        gs[i] = gv .* v .- v .* α
    end
    gμ_prior = grad_logprior(p.prior_mu, μs, Σ)
    return [ g + gp for (g, gp) in zip(gs, gμ_prior) ]
end

function euclid_grad_Sigma(p::HierarchicalRMALAProblem, μs, Σ)
    return euclid_grad_logprior(p.prior_mu, μs, Σ) .+
           grad_logprior(p.prior_sigma, Σ)
end

riemannian_grad_Sigma(p, μs, Σ) = Σ * euclid_grad_Sigma(p, μs, Σ) * Σ

# ----------------------------------------------------------------------------
# — manifold ops
# ----------------------------------------------------------------------------
exp_map_Sigma(Σ, V) = begin
    S  = sqrt(Σ); iS = inv(S)
    return S * exp(iS * V * iS) * S
end

log_map_Sigma(Σ, T) = begin
    S  = sqrt(Σ); iS = inv(S)
    return S * log(iS * T * iS) * S
end

metric_inner_Sigma(Σ, U, V) = tr(inv(Σ)*U*inv(Σ)*V)

function rand_tangent_Sigma(Σ, τ)
    K = size(Σ,1)
    W = randn(K,K); symW = (W + W')/√2; S = sqrt(Σ)
    return √τ * (S * symW * S)
end

log_q_mu(μ₀, μ₁, g₀, τ) = begin Δ = μ₁ .- μ₀ .- (τ/2)*g₀; -dot(Δ,Δ)/(2τ) end
log_q_Sigma(Σ₀, Σ₁, g₀, τ) = begin
    V = log_map_Sigma(Σ₀, Σ₁); Δ = V .- (τ/2)*g₀
    -metric_inner_Sigma(Σ₀, Δ, Δ)/(2τ)
end

# ----------------------------------------------------------------------------
# — RMALA iteration
# ----------------------------------------------------------------------------
function rmala_step(p::HierarchicalRMALAProblem, μs, Σ, τ)
    gμ  = grad_log_post_mu(p, μs, Σ)
    gΣ  = riemannian_grad_Sigma(p, μs, Σ)

    μs′ = [ μ .+ (τ/2)*g .+ √τ*randn(length(μ)) for (μ,g) in zip(μs, gμ) ]

    ζΣ    = rand_tangent_Sigma(Σ, τ)
    Σ′    = exp_map_Sigma(Σ, (τ/2)*gΣ .+ ζΣ)

    lp0   = log_posterior(p, μs, Σ)
    lp1   = log_posterior(p, μs′, Σ′)

    lqf   = sum(log_q_mu(μ, μp, g, τ) for (μ,μp,g) in zip(μs, μs′, gμ)) +
            log_q_Sigma(Σ, Σ′, gΣ, τ)

    gμ′   = grad_log_post_mu(p, μs′, Σ′)
    gΣ′   = riemannian_grad_Sigma(p, μs′, Σ′)

    lqr   = sum(log_q_mu(μp, μ, gp, τ) for (μ,μp,gp) in zip(μs, μs′, gμ′)) +
            log_q_Sigma(Σ′, Σ, gΣ′, τ)

    logα = lp1 - lp0 + (lqr - lqf)
    if log(rand()) < logα
        return μs′, Σ′, true
    else
        return μs, Σ, false
    end
end

# ----------------------------------------------------------------------------
# — full sampler (separate returns)
# ----------------------------------------------------------------------------
function rmala_sampler(p::HierarchicalRMALAProblem,
                       μs0::Vector{Vector{Float64}},
                       Σ0::Matrix{Float64};
                       τ::Float64, num_samples::Int, burnin::Int=0)
    μs, Σ = deepcopy(μs0), deepcopy(Σ0)
    mus_samples   = Vector{Vector{Vector{Float64}}}()
    sigma_samples = Vector{Matrix{Float64}}()
    acc = 0
    for i in 1:(num_samples+burnin)
        μs, Σ, ok = rmala_step(p, μs, Σ, τ)
        acc += ok
        if i > burnin
            push!(mus_samples, deepcopy(μs))
            push!(sigma_samples, deepcopy(Σ))
        end
    end
    @info "Acceptance rate = $(acc/(num_samples+burnin))"
    return mus_samples, sigma_samples
end
