module TestExamples2DHeat

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Heat equation" begin
  @testset "elixir_heat_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_heat_basic.jl"),
      l2   = [1.2700756869568872e-8],
      linf = [6.629198634477973e-8])
  end

  @testset "elixir_heat_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_heat_amr.jl"),
      l2   = [0.08891001605654104],
      linf = [0.6575322065283009])
  end

  @testset "elixir_heat_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_heat_nonperiodic.jl"),
      l2   = [2.050593060856897e-8],
      linf = [6.631714299464696e-8])
  end

  @testset "elixir_heat_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_heat_source_terms.jl"),
      l2   = [0.025350269628084208],
      linf = [0.09030986815695541])
  end
end

end # module
