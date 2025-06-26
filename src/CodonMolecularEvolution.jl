module CodonMolecularEvolution

using FASTX, MolecularEvolution, StatsBase, Distributions, DataFrames, CSV, NLopt, ParameterHandling, LinearAlgebra, LaTeXStrings, Random
using NNlib, Distributions,SimpleUnPack, AbstractMCMC, Interpolations, MCMCChains
using PDMats, BenchmarkTools
using EllipticalSliceSampling
using KrylovKit
using JSON3
abstract type difFUBARGrid end
struct PlotsExtDummy end

include("shared/shared.jl")
include("shared/emptyplot.jl")
include("difFUBAR/difFUBAR.jl")
include("difFUBAR/grids.jl")
include("../test/benchmark_difFUBAR.jl")

include("FUBAR/FUBAR.jl")
include("simulations/alphabeta/alphabeta.jl")
include("simulations/ou_hb.jl")
include("FUBAR/gaussianFUBAR.jl")
include("FUBAR/grid_utilities.jl")
include("simulations/coalescence_scenarios.jl")
include("simulations/alignment_simulations.jl")
export 
    difFUBARBaseline,
    difFUBARParallel,
    difFUBARTreesurgery,
    difFUBARTreesurgeryAndParallel,
    FUBAR,
    dNdS,
    HBdNdS,
    std2maxdNdS,
    maxdNdS2std,
    HB98AA_matrix,
    ShiftingHBSimModel,
    ShiftingNeHBSimModel,
    ShiftingHBSimPartition,
    ShiftingNeHBSimPartition,
    PiecewiseOUModel,
    shiftingHBviz,
    shiftingNeHBviz,
    HBviz,
    FUBAR_analysis,
    FUBAR_tabulate_from_θ,
    SKBDIFUBAR,
    alphabetagrid,
    DirichletFUBAR,
    FIFEFUBAR,
    FUBARGrid,
    simulate_alignment,
    simulate_k_diversifying_sites,
    save_simulation_data,
    # Coalescence scenarios
    CoalescenceScenario,
    HIVScenario,
    InfluenzaScenario,
    COVIDScenario,
    LogisticGrowthScenario,
    SeasonalScenario,
    LogisticScenario,
    effective_population_size,
    sampling_rate, 
    load_scenario
end
