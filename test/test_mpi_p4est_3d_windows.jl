module TestExamplesMPIP4estMesh3D

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "p4est_3d_dgsem")

# NOTE: This file contains a very reduced set of tests for P4estMesh in 3D with MPI on Windows.
#       The reason is that CI using GitHub Actions works very unreliably for the regular test
#       set. While this is by no means sufficient to fully test P4est on Windows in 3D and in
#       parallel, it is at least a smoke test and better than nothing.

@testset "P4estMesh MPI 3D" begin

# Run basic tests
@testset "Examples 3D" begin
  # Linear scalar advection
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      trees_per_dimension=(2 ,2, 2),
      l2   = [0.0025556372420940518],
      linf = [0.016389041111768865])
  end
end

end # P4estMesh MPI

end # module
