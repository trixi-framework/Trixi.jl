module TestExamples1DEulerMulti

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_1d_dgsem")

@testset "Compressible Euler Multicomponent" begin

  @trixi_testset "elixir_eulermulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_ec.jl"),
      l2   = [1.54891912e-01, 4.45430525e-01, 1.70222013e-02, 3.40444026e-02, 6.80888053e-02],
      linf = [2.71276498e-01, 9.95140742e-01, 3.93069410e-02, 7.86138820e-02, 1.57227764e-01])
  end

  @trixi_testset "elixir_eulermulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_es.jl"),
      l2   = [1.53387916e-01, 4.41585576e-01, 3.93605635e-02, 7.87211270e-02],
      linf = [2.49632117e-01, 7.21088064e-01, 6.38328770e-02, 1.27665754e-01])
  end

  @trixi_testset "elixir_eulermulti_convergence_ec.jl" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_ec.jl"),
      l2   = [8.57523604e-05, 1.63878043e-04, 1.94126993e-05, 3.88253986e-05],
      linf = [3.05932773e-04, 6.24480393e-04, 7.25312144e-05, 1.45062429e-04])
  end

  @trixi_testset "elixir_eulermulti_convergence_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_es.jl"),
      l2   = [1.89839338e-05, 6.20774430e-05, 1.54662058e-06, 3.09324115e-06, 6.18648230e-06, 1.23729646e-05],
      linf = [1.20143726e-04, 3.31320722e-04, 6.50836791e-06, 1.30167358e-05, 2.60334716e-05, 5.20669433e-05])
  end

  @trixi_testset "elixir_eulermulti_two_interacting_blast_waves.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_two_interacting_blast_waves.jl"),
      l2   = [1.28886761e+00, 8.27133526e+01, 3.50680272e-03, 1.36987844e-02, 1.91795185e-02],
      linf = [2.96413045e+01, 1.32258448e+03, 9.19191937e-02, 3.10929710e-01, 4.41798976e-0],
      tspan = (0.0, 0.0001))
  end


end



end # module
