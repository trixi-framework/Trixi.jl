module TestExamplesTreeMesh2DPart3

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "TreeMesh2D Part 3" begin
#! format: noindent

# Run basic tests
@testset "Examples 2D" begin
    # MHD
    include("test_tree_2d_mhd.jl")

    # MHD Multicomponent
    include("test_tree_2d_mhdmulti.jl")

    # MHD Multi-ion
    include("test_tree_2d_mhdmultiion.jl")

    # Lattice-Boltzmann
    include("test_tree_2d_lbm.jl")

    # FDSBP methods on the TreeMesh
    include("test_tree_2d_fdsbp.jl")
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)
end # TreeMesh2D Part 3

end #module
