module TestExamples2DAdvDiff

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Scalar advection-diffusion" begin
  @testset "elixir_advdiff_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advdiff_basic.jl"),
      l2   = [6.707801888154751e-5],
      linf = [0.0005912932758251888])
  end
end

end # module
