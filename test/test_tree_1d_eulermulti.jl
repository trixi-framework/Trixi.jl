module TestExamples1DEulerMulti

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi.jl/src/Trixi.jl, dirname gives the parent directory
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
      l2   = [1.8983933794407234e-5, 6.207744299844731e-5, 1.5466205761868047e-6, 3.0932411523736094e-6, 6.186482304747219e-6, 1.2372964609494437e-5],
      linf = [0.00012014372605895218, 0.0003313207215800418, 6.50836791016296e-6, 1.301673582032592e-5, 2.603347164065184e-5, 5.206694328130368e-5])
  end

  @trixi_testset "elixir_eulermulti_convergence_es.jl with flux_chandrashekar" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_es.jl"),
      l2   = [1.88845048e-05, 5.49106005e-05, 9.42673716e-07, 1.88534743e-06, 3.77069486e-06, 7.54138973e-06],
      linf = [1.16223512e-04, 3.07922197e-04, 3.21774233e-06, 6.43548465e-06, 1.28709693e-05, 2.57419386e-05],
      volume_flux = flux_chandrashekar)
  end

  @trixi_testset "elixir_eulermulti_two_interacting_blast_waves.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_two_interacting_blast_waves.jl"),
      l2   = [1.28886761e+00, 8.27133526e+01, 3.50680272e-03, 1.36987844e-02, 1.91795185e-02],
      linf = [2.96413045e+01, 1.32258448e+03, 9.19191937e-02, 3.10929710e-01, 4.41798976e-01],
      tspan = (0.0, 0.0001))
  end


end



end # module
