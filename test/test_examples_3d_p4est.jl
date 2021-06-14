module TestExamples3dP4est

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "3d")

@testset "P4estMesh" begin
  @testset "elixir_advection_basic_p4est.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_p4est.jl"),
      l2   = [0.00013446460962856976],
      linf = [0.0012577781391462928])
  end

  @testset "elixir_advection_p4est_non_conforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_p4est_non_conforming.jl"),
      l2   = [0.0024774648310858928],
      linf = [0.021727876954353964])
  end

  @testset "elixir_advection_amr_p4est.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_p4est.jl"),
      l2   = [9.773852895157622e-6],
      linf = [0.0005853874124926162])
  end
end

end # module
