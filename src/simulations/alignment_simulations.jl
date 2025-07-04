# Enhanced SimulationResult to store sampler metadata

# Abstract type for rate sampling strategies
abstract type RateSampler <: Sampleable{Univariate, Continuous} end

# Base rate samplers (just handle alpha/beta sampling)
struct UnivariateRateSampler <: RateSampler
    alpha_dist::Distribution
    beta_dist::Distribution
end

struct BivariateRateSampler <: RateSampler
    rate_dist::Distribution  # Must be bivariate
end

# Site selection samplers (handle which sites get which rates)
struct AllSitesSampler <: RateSampler
    base_sampler::Union{UnivariateRateSampler, BivariateRateSampler}
end

struct DiversifyingSitesSampler <: RateSampler
    base_sampler::Union{UnivariateRateSampler, BivariateRateSampler}
    diversifying_sites::Int
    total_sites::Int
end
function DiversifyingSitesSampler(base_sampler::Union{UnivariateRateSampler, BivariateRateSampler}, diversifying_site_share::Float64, total_sites::Int)
    return DiversifyingSitesSampler(base_sampler, Int(floor(diversifying_site_share*total_sites)),total_sites)
end

struct SimulationResult 
    grid::Union{FUBARGrid, Nothing}
    tree
    nucs
    nuc_names
    alphavec
    betavec
    scenario::Union{CoalescenceScenario, Nothing}
    rate_sampler::Union{RateSampler, Nothing}
end

function total_branch_length(tree::FelNode) 
    return sum([n.branchlength for n in getnodelist(tree)]) 
end

# Required methods for Sampleable interface
function Base.rand(sampler::UnivariateRateSampler, n::Int)
    alpha_vec = rand(sampler.alpha_dist, n)
    beta_vec = rand(sampler.beta_dist, n)
    return (alpha_vec, beta_vec)
end

function Base.rand(sampler::BivariateRateSampler, n::Int)
    if length(sampler.rate_dist) != 2
        error("Rate distribution must be bivariate")
    end
    rates = rand(sampler.rate_dist, n)
    return (rates[1, :], rates[2, :])
end

function Base.rand(sampler::AllSitesSampler, n::Int)
    return rand(sampler.base_sampler, n)
end

# Helper function to sample individual values for rejection sampling
function sample_individual_values(sampler::UnivariateRateSampler)
    return rand(sampler.alpha_dist), rand(sampler.beta_dist)
end

function sample_individual_values(sampler::BivariateRateSampler)
    rates = rand(sampler.rate_dist)
    return rates[1], rates[2]
end

function Base.rand(sampler::DiversifyingSitesSampler, n::Int)
    if n != sampler.total_sites
        error("DiversifyingSitesSampler expects exactly $(sampler.total_sites) sites")
    end
    
    # Get base rates from the underlying sampler
    alpha_vec, beta_vec = rand(sampler.base_sampler, n)
    
    # Select which sites should be diversifying
    diversifying_indices = shuffle(1:n)[1:sampler.diversifying_sites]
    
    # Apply rejection sampling to ensure correct relationships
    for i in 1:n
        if i ∈ diversifying_indices
            # Ensure beta > alpha for diversifying sites
            while beta_vec[i] <= alpha_vec[i]
                alpha_vec[i], beta_vec[i] = sample_individual_values(sampler.base_sampler)
            end
        else
            # Ensure alpha >= beta for non-diversifying sites
            while alpha_vec[i] < beta_vec[i]
                alpha_vec[i], beta_vec[i] = sample_individual_values(sampler.base_sampler)
            end
        end
    end
    
    return (alpha_vec, beta_vec)
end

function AllSitesSampler(base_sampler::UnivariateRateSampler)
    return Base.@invoke AllSitesSampler(
               base_sampler::Union{UnivariateRateSampler,BivariateRateSampler})
end

function AllSitesSampler(base_sampler::BivariateRateSampler)
    return Base.@invoke AllSitesSampler(
               base_sampler::Union{UnivariateRateSampler,BivariateRateSampler})
end

function DiversifyingSitesSampler(
            base_sampler::UnivariateRateSampler,
            diversifying_sites::Int,
            total_sites::Int)
    if diversifying_sites > total_sites
        error("Number of diversifying sites cannot exceed total sites")
    end
    return Base.@invoke DiversifyingSitesSampler(
               base_sampler::Union{UnivariateRateSampler,BivariateRateSampler},
               diversifying_sites::Int, total_sites::Int)
end

function DiversifyingSitesSampler(
            base_sampler::BivariateRateSampler,
            diversifying_sites::Int,
            total_sites::Int)
    if diversifying_sites > total_sites
        error("Number of diversifying sites cannot exceed total sites")
    end
    return Base.@invoke DiversifyingSitesSampler(
               base_sampler::Union{UnivariateRateSampler,BivariateRateSampler},
               diversifying_sites::Int, total_sites::Int)
end

# Single unified simulation method
function simulate_alignment(ntaxa, scenario::CoalescenceScenario, rate_sampler::RateSampler, nsites, nucleotide_matrix::Array{Float64,2}, f3x4_matrix::Array{Float64,2}; target_normalisation=1.0, create_grid=false)
    alphavec, betavec = rand(rate_sampler, nsites)
    
    # Create wrapper functions that capture the scenario
    Ne_func(t) = effective_population_size(scenario, t)
    sample_rate_func(t) = sampling_rate(scenario, t)
    
    tree = sim_tree(ntaxa, Ne_func, sample_rate_func)
    tbl = total_branch_length(tree)
    for n in getnodelist(tree)
        n.branchlength *= (target_normalisation / tbl)
    end
    # I think scale_total_tree_neutral_expected_subs=1 does what we want it to do?
    nucs, nuc_names, tre = sim_alphabeta_seqs(alphavec, betavec, tree, nucleotide_matrix, f3x4_matrix, scale_total_tree_neutral_expected_subs=target_normalisation)
    nucs, nuc_names, newick_tre = nucs, nuc_names, newick(tre)

    # Conditionally create grid
    grid = create_grid ? alphabetagrid(nuc_names, nucs, newick_tre) : nothing
    
    return SimulationResult(grid, tre, nucs, nuc_names, alphavec, betavec, scenario, rate_sampler)
end

function simulate_alignment(ntaxa,scenario::StandardLadderScenario, rate_sampler::RateSampler, nsites, nucleotide_matrix::Array{Float64,2},f3x4_matrix::Array{Float64,2}; target_normalisation=1.0, create_grid = false)
    alphavec, betavec = rand(rate_sampler, nsites)
    
    tree = ladder_tree_sim(ntaxa)
    tbl = total_branch_length(tree)
    for n in getnodelist(tree)
        n.branchlength *= (target_normalisation / tbl)
    end
    # I think scale_total_tree_neutral_expected_subs=1 does what we want it to do?
    nucs, nuc_names, tre = sim_alphabeta_seqs(alphavec, betavec, tree, nucleotide_matrix, f3x4_matrix, scale_total_tree_neutral_expected_subs=target_normalisation)
    nucs, nuc_names, newick_tre = nucs, nuc_names, newick(tre)

    # Conditionally create grid
    grid = create_grid ? alphabetagrid(nuc_names, nucs, newick_tre) : nothing
    
    return SimulationResult(grid, tre, nucs, nuc_names, alphavec, betavec, scenario, rate_sampler)
end

function simulate_alignment(ntaxa, Ne, sample_rate, rate_sampler::RateSampler, nsites, nucleotide_matrix::Array{Float64,2}, f3x4_matrix::Array{Float64,2}; target_normalisation=1.0, create_grid=false)
    alphavec, betavec = rand(rate_sampler, nsites)
    
    tree = sim_tree(ntaxa, Ne, sample_rate)
    total_branch_length = sum([n.branchlength for n in getnodelist(tree)])
    for n in getnodelist(tree)
        n.branchlength *= (target_normalisation / total_branch_length)
    end
    nucs, nuc_names, tre = sim_alphabeta_seqs(alphavec, betavec, tree, nucleotide_matrix, f3x4_matrix)
    nucs, nuc_names, newick_tre = nucs, nuc_names, newick(tre)

    # Conditionally create grid
    grid = create_grid ? alphabetagrid(nuc_names, nucs, newick_tre) : nothing

    return SimulationResult(grid, tre, nucs, nuc_names, alphavec, betavec, nothing, rate_sampler)
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
    if !isnothing(res.rate_sampler) save_sampler_metadata(res.rate_sampler, name*"_sampler.json") end
end

# Function to save sampler metadata - handles file writing
function save_sampler_metadata(sampler::RateSampler, filename::String)
    metadata = serialize_sampler_to_dict(sampler)
    
    # Save to JSON file
    if !isempty(filename)
        open(filename, "w") do f
            JSON3.write(f, metadata)
        end
    end
    
    return metadata
end

# Multiple dispatch for serialization only
function serialize_sampler_to_dict(sampler::RateSampler)
    return Dict{String, Any}("sampler_type" => string(typeof(sampler)))
end

function serialize_sampler_to_dict(sampler::UnivariateRateSampler)
    metadata = Dict{String, Any}()
    metadata["sampler_type"] = string(typeof(sampler))
    metadata["alpha_distribution"] = string(typeof(sampler.alpha_dist))
    metadata["beta_distribution"] = string(typeof(sampler.beta_dist))
    
    metadata["alpha_params"] = Distributions.params(sampler.alpha_dist)
    metadata["beta_params"] = Distributions.params(sampler.beta_dist)

    
    return metadata
end

function serialize_sampler_to_dict(sampler::BivariateRateSampler)
    metadata = Dict{String, Any}()
    metadata["sampler_type"] = string(typeof(sampler))
    metadata["rate_distribution"] = string(typeof(sampler.rate_dist))
    
    try
        metadata["rate_params"] = Distributions.params(sampler.rate_dist)
    catch
        metadata["rate_params"] = "unknown"
    end
    
    return metadata
end

# Helper function to serialize base sampler
function serialize_base_sampler(sampler::UnivariateRateSampler)
    return serialize_sampler_to_dict(sampler)
end

function serialize_base_sampler(sampler::BivariateRateSampler)
    return serialize_sampler_to_dict(sampler)
end

function serialize_sampler_to_dict(sampler::DiversifyingSitesSampler)
    metadata = Dict{String, Any}()
    metadata["sampler_type"] = string(typeof(sampler))
    metadata["diversifying_sites"] = sampler.diversifying_sites
    metadata["total_sites"] = sampler.total_sites
    
    # Serialize base sampler using multiple dispatch
    base_metadata = serialize_base_sampler(sampler.base_sampler)
    for (key, value) in base_metadata
        metadata["base_$(key)"] = value
    end
    
    return metadata
end

function serialize_sampler_to_dict(sampler::AllSitesSampler)
    metadata = Dict{String, Any}()
    metadata["sampler_type"] = string(typeof(sampler))
    
    # Serialize base sampler using multiple dispatch
    base_metadata = serialize_base_sampler(sampler.base_sampler)
    for (key, value) in base_metadata
        metadata["base_$(key)"] = value
    end
    
    return metadata
end

# Function that computes the total number of expected substitutions under the codon model, when α = β = 1
function compute_total_diversity(res::SimulationResult; nucleotide_matrix = CodonMolecularEvolution.demo_nucmat, f3x4_matrix = CodonMolecularEvolution.demo_f3x4)
    total_substitution_rate = Q_matrix_normalisation_constant(nucleotide_matrix, f3x4_matrix)
    expected_substitutions = total_substitution_rate*total_branch_length(res.tree)
    return expected_substitutions
end
# Computes the "normalising constant" that makes a Q matrix have E[substitutions] = 1 under strict neutrality (α = β = 1)
function Q_matrix_normalisation_constant(nucleotide_matrix, f3x4_matrix)
    π_eq, Q = π_neutral_Q(nucleotide_matrix, f3x4_matrix)
    return -sum(π_eq .* diag(Q))
end
# Computes the Q matrix and equilibrium distribution under neutrality
function π_neutral_Q(nucleotide_matrix, f3x4_matrix)
    π_eq = MolecularEvolution.F3x4_eq_freqs(f3x4_matrix)
    Q = MolecularEvolution.MG94_F3x4(1.0,1.0,nucleotide_matrix, f3x4_matrix)
    return π_eq, Q
end