struct SimulationResult 
    grid::FUBARGrid
    tree
    nucs
    nuc_names
    alphavec
    betavec
end

function simulate_alignment(ntaxa, Ne, sample_rate, alphavec::Vector{Float64}, betavec::Vector{Float64}, nucleotide_matrix::Array{Float64,2}, f3x4_matrix::Array{Float64,2}; target_normalisation=1.0)
    tree = sim_tree(ntaxa, Ne, sample_rate)
    total_branch_length = sum([n.branchlength for n in getnodelist(tree)])
    for n in getnodelist(tree)
        n.branchlength *= (target_normalisation / total_branch_length)
    end
    nucs, nuc_names, tre = sim_alphabeta_seqs(alphavec, betavec, tree, nucleotide_matrix, f3x4_matrix)
    nucs, nuc_names, newick_tre = nucs, nuc_names, newick(tre)

    result = SimulationResult(alphabetagrid(nuc_names, nucs, newick_tre), tre, nucs, nuc_names, alphavec, betavec)
    return result
end
function simulate_alignment(ntaxa, Ne, sample_rate, alpha_distribution::Distribution, beta_distribution::Distribution, nsites, nucleotide_matrix::Array{Float64,2}, f3x4_matrix::Array{Float64,2}; target_normalisation=1.0)
    alphavec = rand(alpha_distribution, nsites)
    betavec = rand(beta_distribution, nsites)
    return simulate_alignment(ntaxa, Ne, sample_rate, alphavec, betavec, nucleotide_matrix, f3x4_matrix, target_normalisation=target_normalisation)
end
function simulate_alignment(ntaxa, Ne, sample_rate, rate_distribution::Distribution, nsites, nucleotide_matrix, f3x4_matrix; target_normalisation=1.0)
    if length(rate_distribution) != 2
        error("Rate distribution must be bivariate")
    end
    rates = rand(rate_distribution, nsites)
    alphavec = rates[1, :]
    betavec = rates[2, :]
    return simulate_alignment(ntaxa, Ne, sample_rate, alphavec, betavec, nucleotide_matrix, f3x4_matrix, target_normalisation=target_normalisation)
end

function rejection_sample(distribution_a::Distribution, distribution_b::Distribution, condition)

    a = rand(distribution_a)
    b = rand(distribution_b)
    while !condition(a,b)
        a = rand(distribution_a)
        b = rand(distribution_b) 
    end
    return a,b
end

function simulate_k_diversifying_sites(ntaxa, Ne, sample_rate, α_distribution::Distribution, β_distribution::Distribution, nsites,diversifying_sites, nucleotide_matrix::Array{Float64,2}, f3x4_matrix::Array{Float64,2}; target_normalisation=1.0)
    diversifying_indices = shuffle(1:nsites)[1:diversifying_sites]
    α_vector = Vector{Float64}()
    β_vector = Vector{Float64}()
    β_vector = []
    for i in 1:nsites
        α = 0
        β = 0
        if i ∈ diversifying_indices
            α, β = rejection_sample(α_distribution, β_distribution, (a,b) -> b > a)
        else 
            α, β = rejection_sample(α_distribution, β_distribution, (a,b) -> a > b)
        end
        push!(α_vector, α)
        push!(β_vector, β)
    end
    return simulate_alignment(ntaxa, Ne, sample_rate, α_vector, β_vector, nucleotide_matrix, f3x4_matrix, target_normalisation = target_normalisation)
end