module TestExamples2DPart2

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "2D-Part2" begin

# Run basic tests
@testset "Examples 2D" begin
  # Compressible Euler Multicomponent
  include("test_examples_2d_eulermulti.jl")

  # MHD
  include("test_examples_2d_mhd.jl")

  # MHD Multicomponent
  include("test_examples_2d_mhdmulti.jl")

  # Lattice-Boltzmann
  include("test_examples_2d_lbm.jl")
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # 2D-Part2

end #module
