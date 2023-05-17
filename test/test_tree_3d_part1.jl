module TestExamplesTreeMesh3DPart1

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "TreeMesh3D Part 1" begin

# Run basic tests
@testset "Examples 3D" begin
  # Compressible Euler
  include("test_tree_3d_euler.jl")
end


# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive=true)

end # TreeMesh3D Part 1

end #module
