# Enhanced SimulationResult to optionally store the scenario and allow grid to be nothing
struct SimulationResult 
    grid::Union{FUBARGrid, Nothing}  # Allow grid to be nothing
    tree
    nucs
    nuc_names
    alphavec
    betavec
    scenario::Union{CoalescenceScenario, Nothing}  # Optional scenario storage
end

function total_branch_length(tree::FelNode) return sum([n.branchlength for n in getnodelist(tree)]) end

# Constructor that defaults grid to nothing when not provided
SimulationResult(grid, tree, nucs, nuc_names, alphavec, betavec) = 
    SimulationResult(grid, tree, nucs, nuc_names, alphavec, betavec, nothing)

# Add new methods that dispatch on CoalescenceScenario
function simulate_alignment(ntaxa, scenario::CoalescenceScenario, alphavec::Vector{Float64}, betavec::Vector{Float64}, nucleotide_matrix::Array{Float64,2}, f3x4_matrix::Array{Float64,2}; target_normalisation=1.0, create_grid=false)
    # Create wrapper functions that capture the scenario
    Ne_func(t) = effective_population_size(scenario, t)
    sample_rate_func(t) = sampling_rate(scenario, t)
    
    return simulate_alignment(ntaxa, Ne_func, sample_rate_func, alphavec, betavec, nucleotide_matrix, f3x4_matrix; target_normalisation=target_normalisation, create_grid=create_grid)
end

function simulate_alignment(ntaxa, scenario::CoalescenceScenario, alpha_distribution::Distribution, beta_distribution::Distribution, nsites, nucleotide_matrix::Array{Float64,2}, f3x4_matrix::Array{Float64,2}; target_normalisation=1.0, create_grid=false)
    alphavec = rand(alpha_distribution, nsites)
    betavec = rand(beta_distribution, nsites)
    return simulate_alignment(ntaxa, scenario, alphavec, betavec, nucleotide_matrix, f3x4_matrix; target_normalisation=target_normalisation, create_grid=create_grid)
end

function simulate_alignment(ntaxa, scenario::CoalescenceScenario, rate_distribution::Distribution, nsites, nucleotide_matrix, f3x4_matrix; target_normalisation=1.0, create_grid=false)
    if length(rate_distribution) != 2
        error("Rate distribution must be bivariate")
    end
    rates = rand(rate_distribution, nsites)
    alphavec = rates[1, :]
    betavec = rates[2, :]
    return simulate_alignment(ntaxa, scenario, alphavec, betavec, nucleotide_matrix, f3x4_matrix; target_normalisation=target_normalisation, create_grid=create_grid)
end

function simulate_k_diversifying_sites(ntaxa, scenario::CoalescenceScenario, α_distribution::Distribution, β_distribution::Distribution, nsites, diversifying_sites, nucleotide_matrix::Array{Float64,2}, f3x4_matrix::Array{Float64,2}; target_normalisation=1.0, create_grid=false)
    diversifying_indices = shuffle(1:nsites)[1:diversifying_sites]
    α_vector = Vector{Float64}()
    β_vector = Vector{Float64}()
    
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
    
    return simulate_alignment(ntaxa, scenario, α_vector, β_vector, nucleotide_matrix, f3x4_matrix; target_normalisation=target_normalisation, create_grid=create_grid)
end

function simulate_alignment(ntaxa, scenario::CoalescenceScenario, alphavec::Vector{Float64}, betavec::Vector{Float64}, nucleotide_matrix::Array{Float64,2}, f3x4_matrix::Array{Float64,2}; target_normalisation=1.0, create_grid=false)
    Ne_func(t) = effective_population_size(scenario, t)
    sample_rate_func(t) = sampling_rate(scenario, t)
    
    tree = sim_tree(ntaxa, Ne_func, sample_rate_func)
    tbl = total_branch_length(tree)
    for n in getnodelist(tree)
        n.branchlength *= (target_normalisation / tbl)
    end
    nucs, nuc_names, tre = sim_alphabeta_seqs(alphavec, betavec, tree, nucleotide_matrix, f3x4_matrix)
    nucs, nuc_names, newick_tre = nucs, nuc_names, newick(tre)

    # Conditionally create grid
    grid = create_grid ? alphabetagrid(nuc_names, nucs, newick_tre) : nothing
    
    result = SimulationResult(grid, tre, nucs, nuc_names, alphavec, betavec, scenario)
    return result
end

function simulate_alignment(ntaxa, Ne, sample_rate, alphavec::Vector{Float64}, betavec::Vector{Float64}, nucleotide_matrix::Array{Float64,2}, f3x4_matrix::Array{Float64,2}; target_normalisation=1.0, create_grid=false)
    tree = sim_tree(ntaxa, Ne, sample_rate)
    total_branch_length = sum([n.branchlength for n in getnodelist(tree)])
    for n in getnodelist(tree)
        n.branchlength *= (target_normalisation / total_branch_length)
    end
    nucs, nuc_names, tre = sim_alphabeta_seqs(alphavec, betavec, tree, nucleotide_matrix, f3x4_matrix)
    nucs, nuc_names, newick_tre = nucs, nuc_names, newick(tre)

    # Conditionally create grid
    grid = create_grid ? alphabetagrid(nuc_names, nucs, newick_tre) : nothing

    result = SimulationResult(grid, tre, nucs, nuc_names, alphavec, betavec)
    return result
end

function simulate_alignment(ntaxa, Ne, sample_rate, alpha_distribution::Distribution, beta_distribution::Distribution, nsites, nucleotide_matrix::Array{Float64,2}, f3x4_matrix::Array{Float64,2}; target_normalisation=1.0, create_grid=false)
    alphavec = rand(alpha_distribution, nsites)
    betavec = rand(beta_distribution, nsites)
    return simulate_alignment(ntaxa, Ne, sample_rate, alphavec, betavec, nucleotide_matrix, f3x4_matrix; target_normalisation=target_normalisation, create_grid=create_grid)
end

function simulate_alignment(ntaxa, Ne, sample_rate, rate_distribution::Distribution, nsites, nucleotide_matrix, f3x4_matrix; target_normalisation=1.0, create_grid=false)
    if length(rate_distribution) != 2
        error("Rate distribution must be bivariate")
    end
    rates = rand(rate_distribution, nsites)
    alphavec = rates[1, :]
    betavec = rates[2, :]
    return simulate_alignment(ntaxa, Ne, sample_rate, alphavec, betavec, nucleotide_matrix, f3x4_matrix; target_normalisation=target_normalisation, create_grid=create_grid)
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

function simulate_k_diversifying_sites(ntaxa, Ne, sample_rate, α_distribution::Distribution, β_distribution::Distribution, nsites, diversifying_sites, nucleotide_matrix::Array{Float64,2}, f3x4_matrix::Array{Float64,2}; target_normalisation=1.0, create_grid=false)
    diversifying_indices = shuffle(1:nsites)[1:diversifying_sites]
    α_vector = Vector{Float64}()
    β_vector = Vector{Float64}()
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
    return simulate_alignment(ntaxa, Ne, sample_rate, α_vector, β_vector, nucleotide_matrix, f3x4_matrix; target_normalisation=target_normalisation, create_grid=create_grid)
end

function save_simulation_data(res::SimulationResult; name = "simulation_data")
    write(name*".nwk", newick(res.tree))
    write_fasta(name*".fasta", res.nucs; seq_names=res.nuc_names)
    
    ground_truth_frame = DataFrame(
        alphavec = res.alphavec, 
        betavec = res.betavec, 
        diversifying_ground_truth = res.betavec .> res.alphavec
    )

    CSV.write(name*"_rates.csv", ground_truth_frame)
    if !isnothing(res.scenario) save_scenario(res.scenario, name*"_scenario.json") end
end



# Function that computes the total number of expected substitutions under the codon model
# This function should take in the codon model and branch length, assume the substitution process starts at stationarity etc
function compute_total_diversity(res::SimulationResult;nucleotide_matrix = CodonMolecularEvolution.demo_nucmat, f3x4_matrix = CodonMolecularEvolution.f3x4)
    π_eq = MolecularEvolution.F3x4_eq_freqs(f3x4_matrix)
    Q = MolecularEvolution.MG94_F3x4(1.0,1.0,nucleotide_matrix, f3x4_matrix)
    total_substitution_rate = -sum(π_eq .* diag(Q))
    expected_substitutions = total_substitution_rate*total_branch_length(res.tree)
    return total_substitution_rate, expected_substitutions
end