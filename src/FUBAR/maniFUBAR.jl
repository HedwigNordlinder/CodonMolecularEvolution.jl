###############################################################################
#  Joint RMALA for  (μ, Σ)  with Σ ∈ P(n)  and  μ ∈ ℝⁿ                         #
###############################################################################
using StatsFuns: logsumexp          # for numerically stable softmax/log-lik

# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers on P(n) – identical to earlier                            #
# ─────────────────────────────────────────────────────────────────────────────
expmap_spd(X,V) = (Xh   = sqrt(X); invXh = inv(Xh);   Xh * exp(invXh*V*invXh) * Xh)
logmap_spd(X,Y) = (Xh   = sqrt(X); invXh = inv(Xh);   Xh * log(invXh*Y*invXh) * Xh)

# geometry drift  a(X)=−(n+1)/2 · X   for affine-invariant metric
geo_drift(X) = -(size(X,1)+1)/2 * X

# ─────────────────────────────────────────────────────────────────────────────
#  Problem definition                                                         #
# ─────────────────────────────────────────────────────────────────────────────
struct RMALAProblem
    grid :: FUBARgrid
    Σ0   :: Matrix{Float64}
    ν0   :: Float64
end

# dimensions that are used many times
dim(p::RMALAProblem) = size(p.Σ0,1)

# ─────────────────────────────────────────────────────────────────────────────
#  Log-posterior  and its gradients                                           #
# ─────────────────────────────────────────────────────────────────────────────
"""
    logposterior(p, μ, Σ)

Un-normalised joint log-posterior  log p(data,μ,Σ).
"""
function logposterior(p::RMALAProblem, μ, Σ)

    # likelihood  ℓ = softmax(μ)' * L
    L   = p.grid.cond_lik_matrix
    log_soft = μ .- logsumexp(μ)
    soft     = exp.(log_soft)
    loglik   = sum( log.(soft' * L) )        # summed over alignment sites

    # log-priors (kernels)
    n, ν0 = dim(p), p.ν0
    Σ0inv = inv(p.Σ0)
    logprior_Σ = -0.5*(ν0 + n + 1)*logdet(Σ) - 0.5*tr(Σ0inv * Σ)
    logprior_μ = -0.5 * μ' * inv(Σ) * μ

    return loglik + logprior_Σ + logprior_μ
end

# Euclidean gradient wrt Σ  (kernel of Wishart × Gaussian)
function gradE_logp_Σ(p::RMALAProblem, μ, Σ)
    n, ν0 = dim(p), p.ν0
    Σinv  = inv(Σ)
    return 0.5*(ν0 - n - 1)*Σinv' - 0.5*(μ*μ') - 0.5*inv(p.Σ0)
end

# Euclidean gradient wrt μ  (softmax log-likelihood + Gaussian prior)
function gradE_logp_μ(p::RMALAProblem, μ, Σ)
    L = p.grid.cond_lik_matrix
    K, N = size(L)             # K = dim, N = #sites
    log_soft = μ .- logsumexp(μ)
    soft     = exp.(log_soft)      # K-vector

    # gradient of log-likelihood wrt μ
    gℓ = zeros(K)
    for j in 1:N
        Lj = @view L[:,j]
        denom = logsumexp(log_soft .+ log.(Lj))
        pj    = exp.(log_soft .+ log.(Lj) .- denom)   # posterior over categories
        gℓ .+= pj .- soft * sum(pj)                   # see thesis Appendix A
    end

    # gradient of Gaussian prior
    gprior = -inv(Σ) * μ
    return gℓ + gprior
end

# Riemannian gradient for Σ
gradR_logp_Σ(p, μ, Σ) = Σ * gradE_logp_Σ(p,μ,Σ) * Σ

# ─────────────────────────────────────────────────────────────────────────────
#  One joint RMALA step                                                       #
# ─────────────────────────────────────────────────────────────────────────────
"""
    rmala_step(problem, μ, Σ; τμ=1e-2, τΣ=1e-2)

Return `(newμ, newΣ, accepted::Bool)`.
Step sizes can differ for the Euclidean and SPD parts.
"""
function rmala_step(prob::RMALAProblem, μ, Σ; τμ=1e-2, τΣ=1e-2)

    # ─── current gradients and drifts ───────────────────────────────────────
    gμ  = gradE_logp_μ(prob, μ, Σ)
    gΣR = gradR_logp_Σ(prob, μ, Σ)

    drift_μ  = 0.5 * gμ                     # Euclidean: only ½∇
    drift_Σ  = 0.5 * gΣR + geo_drift(Σ)     # SPD: ½∇ᴿ + geometry drift

    # ─── propose μᶜ  (Euclidean MALA) ────────────────────────────────────────
    ξμ   = randn(length(μ))
    μcand = μ + τμ*drift_μ + sqrt(2τμ)*ξμ

    # ─── propose Σᶜ  (SPD R-MALA) ────────────────────────────────────────────
    n = dim(prob)
    E = randn(n, n);  E = (E+E')/2
    sqrtΣ = sqrt(Σ)
    ξΣ    = sqrtΣ * E * sqrtΣ                    # noise ~ N(0,G⁻¹)

    V  = τΣ*drift_Σ + sqrt(2τΣ)*ξΣ
    Σcand = expmap_spd(Σ, V)

    # ─── log proposal densities  q((μᶜ,Σᶜ)|(μ,Σ)) and reverse ───────────────
    # Euclidean part
    function logq_euc(to, from, drift_from, τ)
        δ = to - from - τ*drift_from
        k = length(from)
        return -0.5*k*log(4π*τ) - dot(δ,δ)/(4τ)
    end

    # SPD part
    function logq_spd(to, from, drift_from, τ)
        V  = logmap_spd(from,to)
        S  = V - τ*drift_from
        invfrom_S = inv(from)*S
        d = n*(n+1)÷2
        quad = tr(invfrom_S*invfrom_S)
        return -0.5*d*log(4π*τ) + 0.5*(n+1)*logdet(from) - quad/(4τ)
    end

    logq_cur2prop = logq_euc(μcand, μ, drift_μ, τμ) +
                    logq_spd(Σcand, Σ, drift_Σ, τΣ)

    # recompute drifts at the proposal for reverse density
    drift_μp = 0.5 * gradE_logp_μ(prob, μcand, Σcand)
    drift_Σp = 0.5 * gradR_logp_Σ(prob, μcand, Σcand) + geo_drift(Σcand)

    logq_prop2cur = logq_euc(μ, μcand, drift_μp, τμ) +
                    logq_spd(Σ, Σcand, drift_Σp, τΣ)

    # ─── acceptance ratio ───────────────────────────────────────────────────
    logα = logposterior(prob, μcand, Σcand) - logposterior(prob, μ, Σ) +
           logq_prop2cur - logq_cur2prop

    if log(rand()) < logα
        return μcand, Σcand, true
    else
        return μ, Σ, false
    end
end
function run_rmala(prob::RMALAProblem,
                   μ0::AbstractVector,
                   Σ0::AbstractMatrix,
                   nsamples::Integer;
                   burnin::Integer      = 1_000,
                   τμ::Real             = 1e-4,
                   τΣ::Real             = 1e-7,
                   progress::Bool       = false)
    dimμ = length(μ0)

    μ_chain = Matrix{Float64}(undef, dimμ, nsamples)
    Σ_chain = Matrix{Float64}[]

    μ, Σ    = copy(μ0), copy(Σ0)
    accepted = 0
    total    = 0

    niter = nsamples + burnin
    showprog = progress ? Progress(niter; desc="RMALA") : nothing

    for k in 1:niter
        μ, Σ, ok = rmala_step(prob, μ, Σ; τμ=τμ, τΣ=τΣ)
        accepted += ok
        total    += 1
        if k > burnin
            idx = k - burnin
            μ_chain[:,idx] = μ
            push!(Σ_chain, copy(Σ))
        end
        progress && next!(showprog)
    end

    stats = (accept_rate = accepted/total,
             τμ          = τμ,
             τΣ          = τΣ,
             burnin      = burnin)

    return μ_chain, Σ_chain, stats
end