module TestExamples2DHeat

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Heat equation" begin
  @testset "elixir_heat_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_heat_basic.jl"),
      l2   = [0.0008071867035363602],
      linf = [0.004961971226386641])
  end

  @testset "elixir_heat_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_heat_amr.jl"),
      l2   = [0.08891001607226263],
      linf = [0.657532205699553])
  end

  @testset "elixir_heat_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_heat_nonperiodic.jl"),
      l2   = [0.0012846393772784545],
      linf = [0.005583936479563221])
  end
end

end # module
