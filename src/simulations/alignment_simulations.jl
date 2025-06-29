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
    base_sampler::RateSampler
end

struct DiversifyingSitesSampler <: RateSampler
    base_sampler::RateSampler
    diversifying_sites::Int
    total_sites::Int
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
                new_alpha, new_beta = rand(sampler.base_sampler, 1)
                alpha_vec[i] = new_alpha[1]
                beta_vec[i] = new_beta[1]
            end
        else
            # Ensure alpha >= beta for non-diversifying sites
            while alpha_vec[i] < beta_vec[i]
                new_alpha, new_beta = rand(sampler.base_sampler, 1)
                alpha_vec[i] = new_alpha[1]
                beta_vec[i] = new_beta[1]
            end
        end
    end
    
    return (alpha_vec, beta_vec)
end

# Constructor functions for convenience
function UnivariateRateSampler(alpha_dist::Distribution, beta_dist::Distribution)
    return UnivariateRateSampler(alpha_dist, beta_dist)
end

function BivariateRateSampler(rate_dist::Distribution)
    if length(rate_dist) != 2
        error("Rate distribution must be bivariate")
    end
    return BivariateRateSampler(rate_dist)
end

function AllSitesSampler(base_sampler::RateSampler)
    return AllSitesSampler(base_sampler)
end

function DiversifyingSitesSampler(base_sampler::RateSampler, diversifying_sites::Int, total_sites::Int)
    if diversifying_sites > total_sites
        error("Number of diversifying sites cannot exceed total sites")
    end
    return DiversifyingSitesSampler(base_sampler, diversifying_sites, total_sites)
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
    nucs, nuc_names, tre = sim_alphabeta_seqs(alphavec, betavec, tree, nucleotide_matrix, f3x4_matrix)
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
            JSON.print(f, metadata, 2)
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
    
    try
        metadata["alpha_params"] = params(sampler.alpha_dist)
        metadata["beta_params"] = params(sampler.beta_dist)
    catch
        metadata["alpha_params"] = "unknown"
        metadata["beta_params"] = "unknown"
    end
    
    return metadata
end

function serialize_sampler_to_dict(sampler::BivariateRateSampler)
    metadata = Dict{String, Any}()
    metadata["sampler_type"] = string(typeof(sampler))
    metadata["rate_distribution"] = string(typeof(sampler.rate_dist))
    
    try
        metadata["rate_params"] = params(sampler.rate_dist)
    catch
        metadata["rate_params"] = "unknown"
    end
    
    return metadata
end

function serialize_sampler_to_dict(sampler::DiversifyingSitesSampler)
    metadata = Dict{String, Any}()
    metadata["sampler_type"] = string(typeof(sampler))
    metadata["diversifying_sites"] = sampler.diversifying_sites
    metadata["total_sites"] = sampler.total_sites
    
    # Recursively serialize base sampler
    base_metadata = serialize_sampler_to_dict(sampler.base_sampler)
    for (key, value) in base_metadata
        metadata["base_$(key)"] = value
    end
    
    return metadata
end

function serialize_sampler_to_dict(sampler::AllSitesSampler)
    metadata = Dict{String, Any}()
    metadata["sampler_type"] = string(typeof(sampler))
    
    # Recursively serialize base sampler
    base_metadata = serialize_sampler_to_dict(sampler.base_sampler)
    for (key, value) in base_metadata
        metadata["base_$(key)"] = value
    end
    
    return metadata
end

# Function that computes the total number of expected substitutions under the codon model
function compute_total_diversity(res::SimulationResult; nucleotide_matrix = CodonMolecularEvolution.demo_nucmat, f3x4_matrix = CodonMolecularEvolution.demo_f3x4)
    π_eq = MolecularEvolution.F3x4_eq_freqs(f3x4_matrix)
    Q = MolecularEvolution.MG94_F3x4(1.0,1.0,nucleotide_matrix, f3x4_matrix)
    total_substitution_rate = -sum(π_eq .* diag(Q))
    expected_substitutions = total_substitution_rate*total_branch_length(res.tree)
    return total_substitution_rate, expected_substitutions
end