module TestExamples2DStructured

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Structured Mesh" begin
  @testset "elixir_advection_basic_structured.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_structured.jl"),
      l2   = [9.14468177884088e-6],
      linf = [6.437440532947036e-5])
  end

  @testset "elixir_advection_extended_structured.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended_structured.jl"),
      l2   = [2.8572992120045937e-6],
      linf = [1.764895592104576e-5])
  end

  @testset "elixir_advection_extended_structured.jl with polydeg=4" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended_structured.jl"),
      l2   = [4.9753743571684e-7],
      linf = [1.5044127488206271e-6],
      size = (16, 23),
      polydeg = 4,
      cfl = 1.4)
  end
  
  @testset "elixir_advection_restart_structured.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart_structured.jl"),
      l2   = [1.2148032454832534e-5],
      linf = [6.495644795245781e-5])
  end

  @testset "elixir_euler_source_terms_structured.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_structured.jl"),
      l2   = [8.517808508019351e-7, 1.2350203856098537e-6, 1.2350203856728076e-6, 4.277886946638239e-6],
      linf = [8.357848139128876e-6, 1.0326302096741458e-5, 1.0326302101404394e-5, 4.496194024383726e-5])
  end
end

end # module
