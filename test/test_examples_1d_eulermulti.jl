module TestExamples1DEulerMulti

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_1d_dgsem")

@testset "Compressible Euler Multicomponent" begin

  @trixi_testset "elixir_eulermulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_ec.jl"),
      l2   = [1.54912508e-01, 4.45233361e-01, 1.70123765e-02, 3.40247530e-02, 6.80495060e-02],
      linf = [2.70251235e-01, 9.94752402e-01, 3.93233778e-02, 7.86467556e-02, 1.57293511e-01])
  end

  @trixi_testset "elixir_eulermulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_es.jl"),
      l2   = [1.53378791e-01, 4.41560792e-01, 3.93639477e-02, 7.87278954e-02],
      linf = [2.50203604e-01, 7.19643482e-01, 6.36541480e-02, 1.27308296e-01])
  end

  @trixi_testset "elixir_eulermulti_eoc_ec.jl" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_eoc_ec.jl"),
      l2   = [8.54278172e-05, 1.63112465e-04, 1.87964310e-05, 3.75928621e-05],
      linf = [3.13612234e-04, 5.82916495e-04, 7.05047326e-05, 1.41009465e-04])
  end

  @trixi_testset "elixir_eulermulti_eoc_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_eoc_es.jl"),
      l2   = [1.88845048e-05, 5.49106005e-05, 9.42673716e-07, 1.88534743e-06, 3.77069486e-06, 7.54138973e-06],
      linf = [1.16223512e-04, 3.07922197e-04, 3.21774233e-06, 6.43548465e-06, 1.28709693e-05, 2.57419386e-05])
  end

  @trixi_testset "elixir_eulermulti_two_interacting_blast_waves.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_two_interacting_blast_waves.jl"),
      l2   = [1.28874968e+00, 8.28362140e+01, 3.52901633e-03, 1.37049633e-02, 1.91860960e-02],
      linf = [2.96873461e+01, 1.32285301e+03, 9.04336719e-02, 3.10669595e-01, 4.41422167e-01],
      tspan = (0.0, 0.0001))
  end


end



end # module
