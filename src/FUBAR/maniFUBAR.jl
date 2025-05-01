using StatsFuns: logsumexp          # for numerically stable softmax/log-lik
using LinearAlgebra: cholesky, Symmetric, tr, I, mul!, copy!

# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers on P(n) – unchanged                                     
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
#  Problem definition with precomputations                                   
# ─────────────────────────────────────────────────────────────────────────────
struct RMALAProblem
    grid::FUBARgrid
    Σ0::Matrix{Float64}
    ν0::Float64
    F0::Cholesky{Float64,Matrix{Float64}}
    Σ0inv::Matrix{Float64}
    L::Matrix{Float64}
end

function RMALAProblem(grid::FUBARgrid, Σ0::Matrix{Float64}, ν0::Float64)
    F0 = cholesky(Σ0; check=false)                # one-time factorization
    Σ0inv = F0 \ (F0' \ I)                     # Σ0⁻¹
    L = grid.cond_lik_matrix                     # cache likelihood matrix
    RMALAProblem(grid, Σ0, ν0, F0, Σ0inv, L)
end

dim(p::RMALAProblem) = size(p.Σ0, 1)

# ─────────────────────────────────────────────────────────────────────────────
#  Log-posterior  and its gradients                                           
# ─────────────────────────────────────────────────────────────────────────────
function logposterior(p::RMALAProblem, μ, Σ)
    # data likelihood
    log_soft = μ .- logsumexp(μ)
    soft = exp.(log_soft)
    loglik = sum(log.(soft' * p.L))

    # priors
    n, ν0 = dim(p), p.ν0
    F = cholesky(Σ; check=false)
    logdetΣ = 2 * sum(log, diag(F.U))
    Σinvμ = F \ (F' \ μ)

    logprior_Σ = -0.5 * (ν0 + n + 1) * logdetΣ - 0.5 * tr(p.Σ0inv * Σ)
    logprior_μ = -0.5 * dot(μ, Σinvμ)

    return loglik + logprior_Σ + logprior_μ
end

function gradE_logp_Σ(p::RMALAProblem, μ, Σ)
    n, ν0 = dim(p), p.ν0
    F = cholesky(Σ; check=false)
    Σinv = F \ (F' \ I)
    return 0.5 * (ν0 - n - 1) * Σinv - 0.5*(μ*μ') - 0.5*p.Σ0inv
end

gradR_logp_Σ(p, μ, Σ) = Σ * gradE_logp_Σ(p, μ, Σ) * Σ

function gradE_logp_μ(p::RMALAProblem, μ, Σ)
    K, N = size(p.L)
    log_soft = μ .- logsumexp(μ)
    soft = exp.(log_soft)

    gℓ = zeros(K)
    for j in 1:N
        Lj = @view p.L[:, j]
        lognum = log_soft .+ log.(Lj)
        denom = logsumexp(lognum)
        pj = exp.(lognum .- denom)
        gℓ .+= pj .- soft * sum(pj)
    end
    F = cholesky(Σ; check=false)
    gprior = -(F \ μ)
    return gℓ + gprior
end

# ─────────────────────────────────────────────────────────────────────────────
#  One joint RMALA step with in-place buffers                                
# ─────────────────────────────────────────────────────────────────────────────
function rmala_step(prob::RMALAProblem, μ, Σ; τμ=1e-2, τΣ=1e-2,
                     ξμ=Vector{Float64}(), E=Matrix{Float64}(), tmp_vec=Vector{Float64}())
    K = length(μ)
    n = dim(prob)
    # allocate buffers on first call
    isempty(ξμ) && (ξμ .= randn(K); resize!(ξμ, K))
    isempty(E)  && (E .= randn(n,n); resize!(E, n, n))

    # gradients & drifts
    gμ = gradE_logp_μ(prob, μ, Σ)
    gΣR = gradR_logp_Σ(prob, μ, Σ)

    drift_μ = 0.5 .* gμ
    drift_Σ = 0.5 .* gΣR .+ geo_drift(Σ)

    # propose μᶜ
    randn!(ξμ)
    μcand = μ .+ τμ .* drift_μ
    mul!(μcand, sqrt(2τμ), ξμ, 1, 1)

    # propose Σᶜ
    randn!(E)
    SymE = (E + E')/2
    sqrtΣ = sqrt(Σ)
    ξΣ = sqrtΣ * SymE * sqrtΣ

    V = τΣ .* drift_Σ .+ sqrt(2τΣ) .* ξΣ
    Σcand = expmap_spd(Σ, V)

    # proposal log-densities
    logq_euc(to, from, drift, τ) = begin
        δ = to .- from .- τ .* drift
        -0.5 * length(from) * log(4π*τ) - dot(δ,δ)/(4τ)
    end
    logq_spd(to, from, drift, τ) = begin
        V_ = logmap_spd(from, to)
        S = V_ .- τ .* drift
        d = size(from,1)*(size(from,1)+1) ÷ 2
        F = cholesky(from; check=false)
        Y = F \ (F' \ S)
        -0.5*d*log(4π*τ) + 0.5*(size(from,1)+1)*sum(log,diag(F.U)) - tr(Y*Y)/(4τ)
    end

    logq_c2p = logq_euc(μcand, μ, drift_μ, τμ) + logq_spd(Σcand, Σ, drift_Σ, τΣ)
    # reverse drifts
    drift_μp = 0.5 .* gradE_logp_μ(prob, μcand, Σcand)
    drift_Σp = 0.5 .* gradR_logp_Σ(prob, μcand, Σcand) .+ geo_drift(Σcand)
    logq_p2c = logq_euc(μ, μcand, drift_μp, τμ) + logq_spd(Σ, Σcand, drift_Σp, τΣ)

    # Metropolis-Hastings
    logα = logposterior(prob, μcand, Σcand) - logposterior(prob, μ, Σ) + logq_p2c - logq_c2p
    if log(rand()) < logα
        return μcand, Σcand, true
    else
        return μ, Σ, false
    end
end

# ─────────────────────────────────────────────────────────────────────────────
#  Main sampler                                                             
# ─────────────────────────────────────────────────────────────────────────────
function run_rmala(prob::RMALAProblem,
                   μ0::AbstractVector,
                   Σ0::AbstractMatrix,
                   nsamples::Integer;
                   burnin::Integer=1_000,
                   τμ::Real=1e-4,
                   τΣ::Real=1e-7,
                   progress::Bool=false)
    dimμ = length(μ0)
    μ_chain = Matrix{Float64}(undef, dimμ, nsamples)
    Σ_chain = Matrix{Float64}[]
    μ, Σ = copy(μ0), copy(Σ0)
    accepted = 0; total = 0

    # pre-allocate noise buffers
    ξμ = zeros(dimμ)
    E   = zeros(dim(Σ,1), dim(Σ,1))

    for k in 1:(nsamples + burnin)
        μ, Σ, ok = rmala_step(prob, μ, Σ;
                              τμ=τμ, τΣ=τΣ,
                              ξμ=ξμ, E=E)
        accepted += ok; total += 1
        if k > burnin
            idx = k - burnin
            μ_chain[:, idx] = μ
            push!(Σ_chain, copy(Σ))  # small unavoidable allocation
        end
    end

    stats = (accept_rate = accepted/total, τμ=τμ, τΣ=τΣ, burnin=burnin)
    return μ_chain, Σ_chain, stats
end
