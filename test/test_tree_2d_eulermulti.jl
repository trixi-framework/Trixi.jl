module TestExamples2DEulerMulticomponent

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi.jl/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "Compressible Euler Multicomponent" begin
  # NOTE: Some of the L2/Linf errors are comparably large. This is due to the fact that some of the
  #       simulations are set up with dimensional states. For example, the reference pressure in SI
  #       units is 101325 Pa, i.e., pressure has values of O(10^5)

  @trixi_testset "elixir_eulermulti_shock_bubble.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_shock_bubble.jl"),
      l2   = [73.78467629094177, 0.9174752929795251, 57942.83587826468, 0.1828847253029943, 0.011127037850925347],
      linf = [196.81051991521073, 7.8456811648529605, 158891.88930113698, 0.811379581519794, 0.08011973559187913],
      tspan = (0.0, 0.001))
  end

  @trixi_testset "elixir_eulermulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_ec.jl"),
      l2   = [0.050182236154087095, 0.050189894464434635, 0.2258715597305131, 0.06175171559771687],
      linf = [0.3108124923284472, 0.3107380389947733, 1.054035804988521, 0.29347582879608936])
  end

  @trixi_testset "elixir_eulermulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_es.jl"),
      l2   = [0.0496546258404055, 0.04965550099933263, 0.22425206549856372, 0.004087155041747821, 0.008174310083495642, 0.016348620166991283, 0.032697240333982566],
      linf = [0.2488251110766228, 0.24832493304479406, 0.9310354690058298, 0.017452870465607374, 0.03490574093121475, 0.0698114818624295, 0.139622963724859])
  end

  @trixi_testset "elixir_eulermulti_convergence_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_ec.jl"),
      l2   = [0.00012290225488326508, 0.00012290225488321876, 0.00018867397906337653, 4.8542321753649044e-5, 9.708464350729809e-5],
      linf = [0.0006722819239133315, 0.0006722819239128874, 0.0012662292789555885, 0.0002843844182700561, 0.0005687688365401122])
  end

  @trixi_testset "elixir_eulermulti_convergence_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_es.jl"),
      l2   = [2.2661773867001696e-6, 2.266177386666318e-6, 6.593514692980009e-6, 8.836308667348217e-7, 1.7672617334696433e-6],
      linf = [1.4713170997993075e-5, 1.4713170997104896e-5, 5.115618808515521e-5, 5.3639516094383666e-6, 1.0727903218876733e-5])
  end

  @trixi_testset "elixir_eulermulti_convergence_es.jl with flux_chandrashekar" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_es.jl"),
      l2   = [1.8621737639352465e-6, 1.862173764098385e-6, 5.942585713809631e-6, 6.216263279534722e-7, 1.2432526559069443e-6],
      linf = [1.6235495582606063e-5, 1.6235495576388814e-5, 5.854523678827661e-5, 5.790274858807898e-6, 1.1580549717615796e-5],
      volume_flux = flux_chandrashekar)
  end
end

end # module
