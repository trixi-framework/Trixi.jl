module TestExamplesTreeMesh2DPart2

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "TreeMesh2D Part 2" begin
#! format: noindent

# Run basic tests
@testset "Examples 2D" begin
    # Acoustic perturbation
    include("test_tree_2d_acoustics.jl")

    # Linearized Euler
    include("test_tree_2d_linearizedeuler.jl")

    # Compressible Euler
    include("test_tree_2d_euler.jl")

    # Compressible Euler Multicomponent
    include("test_tree_2d_eulermulti.jl")

    # Compressible Polytropic Euler
    include("test_tree_2d_eulerpolytropic.jl")

    # Compressible Euler coupled with acoustic perturbation equations
    include("test_tree_2d_euleracoustics.jl")

    # KPP problem
    include("test_tree_2d_kpp.jl")
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)
end # TreeMesh2D Part 2

end #module
