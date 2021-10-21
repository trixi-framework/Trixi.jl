module TestExamplesTreeMesh2DPart2

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "TreeMesh2D Part 2" begin

# Run basic tests
@testset "Examples 2D" begin
  # Acoustic perturbation
  include("test_tree_2d_acoustics.jl")

  # Compressible Euler
  include("test_tree_2d_euler.jl")

  # Compressible Euler Multicomponent
  include("test_tree_2d_eulermulti.jl")

  # Compressible Euler coupled with acoustic perturbation equations
  include("test_tree_2d_euleracoustics.jl")
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # TreeMesh2D Part 2

end #module
