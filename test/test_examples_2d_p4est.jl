module TestExamples2DCurved

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "P4estMesh" begin
  @testset "elixir_advection_basic_p4est.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_p4est.jl"),
      l2   = [9.14468177884088e-6],
      linf = [6.437440532947036e-5])
  end

  @testset "elixir_advection_restart_p4est.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart_p4est.jl"),
      l2   = [6.398955192910044e-6],
      linf = [3.474337336717426e-5])
  end
end

end # module
