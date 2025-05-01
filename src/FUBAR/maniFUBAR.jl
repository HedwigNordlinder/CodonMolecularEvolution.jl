struct RMALAProblem
    grid::FUBARgrid
    Σ0::Matrix{Float64}
    ν0::Float64
    dim::Int
end

function wishart_logprior_kernel(problem::RMALAProblem, Σ)
    n = problem.dim
    ν = problem.ν0
    Σ0 = problem.Σ0
    return -0.5 * (ν + n + 1) * logdet(Σ) - 0.5 * tr(Σ0 \ Σ)
end

function gaussian_logprior_kernel(problem::RMALAProblem, μ, Σ)
    return -0.5 * (μ' * inv(Σ) * μ)
end

function combined_logprior(problem::RMALAProblem, μ, Σ)
    return wishart_logprior_kernel(problem, Σ) + gaussian_logprior_kernel(problem, μ, Σ)
end

function logposterior(grid::FUBARgrid,μ, Σ)
    ℓ = sum(log.(softmax(μ)'grid.cond_lik_matrix))
    return ℓ + combined_logprior(problem, μ, Σ)
end

# Euclidean gradient wrt Σ
function dℓdΣ(problem::RMALAProblem, Σ, μ)
    Σ_inv_T = transpose(inv(Σ))
    μ_outer = μ * transpose(μ)
    V_inv = inv(problem.Σ0)
    
    result = ((problem.ν0 - problem.dim - 2)/2) * Σ_inv_T - 0.5 * μ_outer - 0.5 * V_inv
    return result
end

function dℓdμ(problem::RMALAProblem, μ, Σ)
    K = problem.dim
    N = size(problem.grid.cond_lik_matrix, 2)
    M = problem.grid.cond_lik_matrix 
    log_p = μ .- logsumexp(μ)
    p = exp.(log_p)
    ∇ℓ = zeros(K)
    for j in 1:N
        M_j = @view M[:, j]
        denom = logsumexp(log_p .+ log.(M_j)) 
        term1 = p .* (M_j ./ exp(denom))       
        term2 = p * sum(p .* (M_j ./ exp(denom))) 
        ∇ℓ .+= term1 .- term2
    end
    ∇prior = -inv(Σ) * μ
    return ∇ℓ + ∇prior
end

function riemannian_dℓdΣ(problem::RMALAProblem, Σ, μ)
    return Σ*dℓdΣ(problem, Σ, μ)*Σ
end



function rmala_step(problem::RMALAProblem, X, ν, Σ; τ=1e-2, rng=Random.GLOBAL_RNG)

    n   = problem.dim
    d   = n*(n+1)÷2                           # tangent dimension

    # total drift   m(X)=½∇ᴿlog p + a(X)
    drift = 0.5*gradR_logp(X,ν,Σ) + geo_drift(X)

    # isotropic noise ξ ~ N(0, G⁻¹)
    E = randn(rng,n,n);  E = (E+E')/2
    sqrtX = sqrt(X)
    ξ = sqrtX * E * sqrtX

    # proposal in tangent & exponential map
    V = τ*drift + sqrt(2τ)*ξ
    Y = expmap_spd(X,V)

    # log proposal density  q(Y|X)
    function logq(to,from,drift_from)
        V   = logmap_spd(from,to)
        S   = V - τ*drift_from
        invfrom_S = inv(from)*S
        quad   = tr(invfrom_S*invfrom_S)            # ||S||²_from
        return  -0.5*d*log(4π*τ) + 0.5*(n+1)*logdet(from) - quad/(4τ)
    end

    driftX = drift
    driftY = 0.5*gradR_logp(Y,ν,Σ) + geo_drift(Y)

    logα = logp(Y,ν,Σ) - logp(X,ν,Σ) +
           logq(X,Y,driftY) - logq(Y,X,driftX)

    if log(rand(rng)) < logα
        return Y,true
    else
        return X,false
    end
end