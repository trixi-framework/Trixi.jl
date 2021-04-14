module TestExamples3DCurved

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "3d")

@testset "Curved mesh" begin
  @testset "elixir_advection_basic_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_curved.jl"),
      l2   = [0.00013446460962856976],
      linf = [0.0012577781391462928])
  end

  @testset "elixir_advection_free_stream_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_free_stream_curved.jl"),
      l2   = [1.830875777528287e-14],
      linf = [1.525446435834965e-12])
  end

  @testset "elixir_euler_source_terms_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_curved.jl"),
      l2   = [0.01032310150257373, 0.009728768969448439, 0.009728768969448494, 0.009728768969448388, 0.015080412597559597],
      linf = [0.034894790428615874, 0.033835365548322116, 0.033835365548322116, 0.03383536554832034, 0.06785765131417065])
  end

  @testset "elixir_euler_free_stream_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream_curved.jl"),
      l2   = [2.8815700334367128e-15, 9.361915278236651e-15, 9.95614203619935e-15, 1.6809941842374106e-14, 1.4815037041566735e-14],
      linf = [4.1300296516055823e-14, 2.0444756998472258e-13, 1.0133560657266116e-13, 3.1896707497480747e-13, 6.092903959142859e-13])
  end
end

end # module
