module CodonMolecularEvolution

using FASTX, MolecularEvolution, StatsBase, Distributions, DataFrames, CSV, NLopt, ParameterHandling, LinearAlgebra, LaTeXStrings, Random
using NNlib, Distributions,SimpleUnPack, AbstractMCMC, Interpolations, MCMCChains
using PDMats, BenchmarkTools
using EllipticalSliceSampling
using KrylovKit
abstract type difFUBARGrid end

include("shared/shared.jl")
include("difFUBAR/difFUBAR.jl")
include("difFUBAR/grids.jl")
include("../test/benchmark_difFUBAR.jl")

include("FUBAR/FUBAR.jl")
include("simulations/alphabeta/alphabeta.jl")
include("simulations/ou_hb.jl")
include("FUBAR/gaussianFUBAR.jl")
include("FUBAR/grid_utilities.jl")
include("FUBAR/maniFUBAR_hierarchical.jl")
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
    ShiftingHBSimPartition,
    PiecewiseOUModel,
    shiftingHBviz,
    HBviz,
    FUBAR_analysis,
    SKBDIFUBAR,
    alphabetagrid,
    DirichletFUBAR,
    FIFEFUBAR,
    FUBARgrid,
    RMALAProblem,
    WishartProblem,
    HierarchicalRMALAWishart,
    HierarchicalRMLALKJ,
    HierarchicalRMALARiemGauss,
    run_rmala
end
