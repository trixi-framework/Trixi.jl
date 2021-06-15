module TestExamples1DAdvection

using Test, SafeTestsets
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "1d")

@safetestset "Linear scalar advection" begin
  @safetestset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [6.0388296447998465e-6],
      linf = [3.217887726258972e-5])
  end

  @safetestset "elixir_advection_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
      l2   = [0.3540206249507417],
      linf = [0.9999896603382347])
  end

  @safetestset "elixir_advection_amr_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_nonperiodic.jl"),
      l2   = [4.283508859843524e-6],
      linf = [3.235356127918171e-5])
  end
end

end # module
