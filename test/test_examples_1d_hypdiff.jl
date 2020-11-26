module TestExamples1DHypDiff

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "1d")

@testset "Hyperbolic diffusion" begin

  @testset "elixir_hypdiff_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic.jl"),
      l2   = [1.3655114954641076e-7, 1.0200345025539218e-6],
      linf = [7.173285538342178e-7, 4.507116681651269e-6])
  end

  @testset "elixir_hypdiff_harmonic_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_harmonic_nonperiodic.jl"),
      l2   = [3.8016596717768455e-12, 3.31057031526513e-12],
      linf = [5.115019519053021e-12, 4.900080341485591e-12])
  end
end

end # module
