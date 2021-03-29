module TestExamples1DStructured

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "1d")

@testset "Structured Mesh" begin
  @testset "elixir_advection_basic_structured.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_structured.jl"),
      l2   = [6.0388296447998465e-6],
      linf = [3.217887726258972e-5])
  end
  
  @testset "elixir_euler_source_terms_structured.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_structured.jl"),
    l2   = [2.2527950196212703e-8, 1.8187357193835156e-8, 7.705669939973104e-8],
    linf = [1.6205433861493646e-7, 1.465427772462391e-7, 5.372255111879554e-7])
  end
end

end # module
