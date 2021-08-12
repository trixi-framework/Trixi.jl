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
      l2   = [7.37942029e+01, 8.93064845e-01, 5.79494667e+04, 1.82993538e-01, 1.11657182e-02],
      linf = [1.96909734e+02, 7.50913818e+00, 1.58891250e+05, 8.18676275e-01, 8.01984099e-02],
      tspan = (0.0, 0.001))
  end

  @trixi_testset "elixir_eulermulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_ec.jl"),
      l2   = [5.01776719e-02, 5.01847697e-02, 2.25842834e-01, 6.17662476e-02],
      linf = [3.08624314e-01, 3.08758298e-01, 1.05971143e+00, 3.00220439e-01])
  end

  @trixi_testset "elixir_eulermulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_es.jl"),
      l2   = [4.96549498e-02, 4.96558239e-02, 2.24252879e-01, 4.08716843e-03, 8.17433686e-03, 1.63486737e-02, 3.26973475e-02],
      linf = [2.48866593e-01, 2.48362494e-01, 9.31003945e-01, 1.74513568e-02, 3.49027135e-02, 6.98054270e-02, 1.39610854e-01])
  end

  @trixi_testset "elixir_eulermulti_convergence_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_ec.jl"),
      l2   = [1.22902255e-04, 1.22902255e-04, 1.88673979e-04, 4.85423218e-05, 9.70846435e-05],
      linf = [6.72281924e-04, 6.72281924e-04, 1.26622928e-03, 2.84384418e-04, 5.68768837e-04])
  end

  @trixi_testset "elixir_eulermulti_convergence_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_es.jl"),
      l2   = [1.74779183e-06, 1.74779183e-06, 5.62550396e-06, 6.15682712e-07, 1.23136542e-06],
      linf = [1.51966473e-05, 1.51966473e-05, 5.51065526e-05, 5.58096521e-06, 1.11619304e-05])
  end
end

end # module
