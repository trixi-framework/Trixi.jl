module TestExamples2DPart3

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "2D-Part3" begin

# Run basic tests
@testset "Examples 2D" begin
  # Acoustic perturbation
  include("test_examples_2d_ape.jl")

  # Curved mesh
  include("test_examples_2d_curved.jl")

  # Unstructured curved mesh
  include("test_examples_2d_unstructured_quad.jl")

  # P4estMesh
  include("test_examples_2d_p4est.jl")

  # FDSBP methods on the TreeMesh
  include("test_tree_2d_fdsbp.jl")
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # 2D-Part3

end #module
