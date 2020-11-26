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

end

end # module
