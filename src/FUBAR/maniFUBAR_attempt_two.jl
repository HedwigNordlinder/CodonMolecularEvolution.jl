# ----------------------------------------------------------------------------
# — abstract "Prior" interface + dispatchers
# ----------------------------------------------------------------------------
abstract type Prior end

# — Forward declaration of the problem type
struct HierarchicalRMALAProblem{Pμ<:Prior, PΣ<:Prior}
    grids
    prior_mu::Pμ
    prior_sigma::PΣ
end

# — Normal prior on each μᵢ ~ 𝒩(0, Σ)
struct NormalPrior <: Prior end

function logprior(::NormalPrior, μs::Vector{Vector{T}}, Σ::AbstractMatrix{T}) where T
    lp = 1/2 * logdet(Σ)
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
    return ((ν - K - 1)/2)*logdet(Σ) - 1/2*tr(inv(V0)*Σ)
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
function loglikelihood(p::HierarchicalRMALAProblem, μs)
    ll = 0.0
    for (i, μ) in enumerate(μs)
        L = p.grids[i].cond_lik_matrix      # K×N
        # log-softmax → probabilities v sum to 1
        log_soft = μ .- logsumexp(μ)
        v        = exp.(log_soft)
        M        = v' * L                  # 1×N row vector
        ll      += sum(log.(M))            # sum_j log M_j
    end
    return ll
end

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
        v = exp.(μ .- logsumexp(μ)); v ./= sum(v)
        M = v' * L
        w = vec(1.0 ./ M)
        grad_v = L * w
        α = dot(v, grad_v)
        gs[i] = v .* (grad_v .- α)
    end
    gμ_prior = grad_logprior(p.prior_mu, μs, Σ)
    return [g + gp for (g, gp) in zip(gs, gμ_prior)]
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
# — RMALA iteration with separate taus
# ----------------------------------------------------------------------------
function rmala_step(p::HierarchicalRMALAProblem, μs, Σ, τ_mu::Float64, τ_sigma::Float64)
    # Compute gradients
    gμ  = grad_log_post_mu(p, μs, Σ)
    gΣ  = riemannian_grad_Sigma(p, μs, Σ)

    # Propose new mu with its own step size
    μs_new = [ μ .+ (τ_mu/2)*g .+ sqrt(τ_mu)*randn(length(μ)) for (μ,g) in zip(μs, gμ) ]

    # Propose new Sigma with its own step size
    ζΣ    = rand_tangent_Sigma(Σ, τ_sigma)
    Σ_new = exp_map_Sigma(Σ, (τ_sigma/2)*gΣ .+ ζΣ)

    # Compute log-posterior at old and new
    lp0   = log_posterior(p, μs, Σ)
    lp1   = log_posterior(p, μs_new, Σ_new)

    # Forward proposal densities
    lqf   = sum(log_q_mu(μ, μp, g, τ_mu) for (μ,μp,g) in zip(μs, μs_new, gμ)) +
             log_q_Sigma(Σ, Σ_new, gΣ, τ_sigma)

    # Reverse proposal densities
    gμ_new = grad_log_post_mu(p, μs_new, Σ_new)
    gΣ_new = riemannian_grad_Sigma(p, μs_new, Σ_new)
    lqr   = sum(log_q_mu(μp, μ, gp, τ_mu) for (μ,μp,gp) in zip(μs, μs_new, gμ_new)) +
             log_q_Sigma(Σ_new, Σ, gΣ_new, τ_sigma)

    # Accept/reject
    logα = lp1 - lp0 + (lqr - lqf)
    if log(rand()) < logα
        return μs_new, Σ_new, true
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
                       τ_mu::Float64=1e-4,
                       τ_sigma::Float64=1e-4,
                       num_samples::Int=1000,
                       burnin::Int=0)
    μs, Σ = deepcopy(μs0), deepcopy(Σ0)
    mus_samples   = Vector{Vector{Vector{Float64}}}()
    sigma_samples = Vector{Matrix{Float64}}()
    logpost_samples = Vector{Float64}()
    acc = 0
    total_iter = 0
    current_logpost = log_posterior(p, μs, Σ)

    while length(mus_samples) < num_samples
        total_iter += 1
        μs_new, Σ_new, accepted = rmala_step(p, μs, Σ, τ_mu, τ_sigma)
        if accepted
            acc += 1; μs, Σ = μs_new, Σ_new
            current_logpost = log_posterior(p, μs, Σ)
            if total_iter > burnin
                push!(mus_samples, deepcopy(μs))
                push!(sigma_samples, deepcopy(Σ))
                push!(logpost_samples, current_logpost)
            end
        end
        if total_iter % 100 == 0
            @info "Iter $(total_iter): Acceptance rate = $(acc/total_iter), Collected $(length(mus_samples))/$(num_samples) samples"
        end
    end
    @info "Final acceptance rate = $(acc/total_iter), Total iterations = $(total_iter)"
    return mus_samples, sigma_samples, logpost_samples
end
