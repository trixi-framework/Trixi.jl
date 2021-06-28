module TestExamples2DEulerMulticomponent

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "Compressible Euler Multicomponent" begin
  # NOTE: Some of the L2/Linf errors are comparably large. This is due to the fact that some of the
  #       simulations are set up with dimensional states. For example, the reference pressure in SI
  #       units is 101325 Pa, i.e., pressure has values of O(10^5)

  @trixi_testset "elixir_eulermulti_shock_bubble.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_shock_bubble.jl"),
      l2   = [7.38219722e+01, 8.80169627e-01, 5.79496021e+04, 1.83101743e-01, 1.11794879e-02],
      linf = [1.96834444e+02, 7.19013908e+00, 1.58095793e+05, 8.19793567e-01, 8.00831673e-02],
      tspan = (0.0, 0.001))
  end

  @trixi_testset "elixir_eulermulti_shock_bubble_3components.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_shock_bubble_3components.jl"),
      l2   = [6.69469420e+01, 1.24137635e+00, 5.19736818e+04, 1.67537081e-01, 6.94144768e-03, 7.33198795e-02],
      linf = [2.14068118e+02, 1.82870581e+01, 1.93352191e+05, 1.02911816e+00, 7.00111358e-02, 9.34275213e-01],
      tspan = (0.0, 0.001))
  end

  @trixi_testset "elixir_eulermulti_shock_bubble_7components.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_shock_bubble_7components.jl"),
    l2   = [6.70895845e+01, 1.50857110e+00, 5.35958153e+04, 1.88695032e-01, 5.30253136e-03, 3.89776034e-02, 7.99227155e-02, 3.50734339e-03, 3.06391963e-02, 1.17354254e-01],
    linf = [2.27410552e+02, 1.95446320e+01, 2.96650723e+05, 1.09905431e+00, 4.40228926e-02, 5.08315400e-01, 1.06801398e+00, 1.76102931e-02, 4.01158590e-01, 1.47984635e+00],
    tspan = (0.0, 0.001))
  end

  @trixi_testset "elixir_eulermulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_ec.jl"),
      l2   = [5.01948074e-02, 5.02023248e-02, 2.25886833e-01, 6.17286464e-02],
      linf = [3.06937711e-01, 3.06807092e-01, 1.06295287e+00, 2.98135725e-01])
  end

  @trixi_testset "elixir_eulermulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_es.jl"),
      l2   = [4.96608585e-02, 4.96617550e-02, 2.24265059e-01, 4.08685600e-03, 8.17371200e-03, 1.63474240e-02, 3.26948480e-02],
      linf = [2.48820395e-01, 2.48311640e-01, 9.35790843e-01, 1.72574990e-02, 3.45149980e-02, 6.90299959e-02, 1.38059992e-01])
  end

  @trixi_testset "elixir_eulermulti_convergence_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_ec.jl"),
      l2   = [1.15269058e-04, 1.15269058e-04, 1.66578747e-04, 4.61097337e-05, 9.22194673e-05],
      linf = [5.05618608e-04, 5.05618608e-04, 7.87591809e-04, 2.06476960e-04, 4.12953919e-04])
  end

  @trixi_testset "elixir_eulermulti_convergence_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_es.jl"),
      l2   = [1.74779183e-06, 1.74779183e-06, 5.62550396e-06, 6.15682712e-07, 1.23136542e-06],
      linf = [1.51966473e-05, 1.51966473e-05, 5.51065526e-05, 5.58096521e-06, 1.11619304e-05])
  end
end

end # module
