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
      l2   = [0.050182236154087095, 0.050189894464434635, 0.2258715597305131, 0.06175171559771687],
      linf = [0.3108124923284472, 0.3107380389947733, 1.054035804988521, 0.29347582879608936])
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
      l2   = [2.1855868685251012e-6, 2.1855868686504143e-6, 6.339327237279491e-6, 8.886020745630879e-7, 1.7772041491261757e-6],
      linf = [1.3465800098977354e-5, 1.3465800097645086e-5, 4.7306482458431276e-5, 5.182977349305062e-6, 1.0365954698610125e-5])
  end

  @trixi_testset "elixir_eulermulti_convergence_es.jl with flux_chandrashekar" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_es.jl"),
      l2   = [1.74779183e-06, 1.74779183e-06, 5.62550396e-06, 6.15682712e-07, 1.23136542e-06],
      linf = [1.51966473e-05, 1.51966473e-05, 5.51065526e-05, 5.58096521e-06, 1.11619304e-05],
      volume_flux = flux_chandrashekar)
  end
end

end # module
