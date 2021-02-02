module TestExamples2DAdvDiff

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Scalar advection-diffusion" begin
  @testset "elixir_advdiff_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advdiff_basic.jl"),
      l2   = [6.707802391539727e-5],
      linf = [0.0005912935421852339])
  end

  @testset "elixir_advdiff_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advdiff_amr.jl"),
      l2   = [0.16503674544814984],
      linf = [0.9236893624634693])
  end
end

end # module
