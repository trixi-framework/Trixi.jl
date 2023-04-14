module TestExamples1DMHD

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi.jl/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_1d_dgsem")

@testset "MHD" begin

  @trixi_testset "elixir_mhd_alfven_wave.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [1.440611823425164e-15, 1.1373567770134494e-14, 3.024482376149653e-15, 2.0553143516814395e-15, 3.9938347410210535e-14, 3.984545392098788e-16, 2.4782402104201577e-15, 1.551737464879987e-15],
      linf = [1.9984014443252818e-15, 1.3405943022348765e-14, 3.3584246494910985e-15, 3.164135620181696e-15, 7.815970093361102e-14, 8.881784197001252e-16, 2.886579864025407e-15, 2.942091015256665e-15],
      initial_condition = initial_condition_constant,
      tspan = (0.0,1.0))
  end

  @trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [1.0375628983659061e-5, 6.571144191446236e-7, 3.5833569836289104e-5, 3.583356983615859e-5, 5.084863194951084e-6, 1.1963224165731992e-16, 3.598916927583752e-5, 3.598916927594727e-5],
      linf = [2.614095879338585e-5, 9.577266731216823e-7, 0.00012406198007461344, 0.00012406198007509917, 1.5066209528846741e-5, 2.220446049250313e-16, 0.00012658678753942054, 0.00012658678753908748])
  end

  @trixi_testset "elixir_mhd_alfven_wave.jl with flux_derigs_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [1.4396053943470756e-5, 1.1211016739165248e-5, 3.577870687983967e-5, 3.577870687982181e-5, 1.967962220860377e-6, 1.1963224165731992e-16, 3.583562899483433e-5, 3.583562899486565e-5],
      linf = [5.830577969345718e-5, 3.280495696370357e-5, 0.00012279619948236953, 0.00012279619948227238, 6.978806516122482e-6, 2.220446049250313e-16, 0.00012564003648959932, 0.00012564003648994626],
      volume_flux = flux_derigs_etal)
  end

  @trixi_testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [5.86009540e-02, 8.16048158e-02, 5.46791194e-02, 5.46791194e-02, 1.54509265e-01, 4.13046273e-17, 5.47637521e-02, 5.47637521e-02],
      linf = [1.10014999e-01, 1.81982581e-01, 9.13611439e-02, 9.13611439e-02, 4.23831370e-01, 1.11022302e-16, 9.93731761e-02, 9.93731761e-02])
  end

  @trixi_testset "elixir_mhd_briowu_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_briowu_shock_tube.jl"),
      l2   = [0.17764301067932906, 0.19693621875378622, 0.3635136528288642, 0.0, 0.3757321708837591, 8.593007507325741e-16, 0.36473438378159656, 0.0],
      linf = [0.5601530250396535, 0.43867368105486537, 1.0960903616351099, 0.0, 1.0551794137886303, 4.107825191113079e-15, 1.5374410890043144, 0.0],
      coverage_override = (maxiters=6,))
  end

  @trixi_testset "elixir_mhd_torrilhon_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_torrilhon_shock_tube.jl"),
      l2   = [0.45700904847931145, 0.4792535936512035, 0.340651203521865, 0.4478034694296928, 0.9204708961093411, 1.3216517820475193e-16, 0.28897419402047725, 0.25521206483145126],
      linf = [1.2185238171352286, 0.8913202384963431, 0.8488793580488431, 0.973083603686, 1.660723397705417, 2.220446049250313e-16, 0.6874726847741993, 0.65536978110274])
  end

  @trixi_testset "elixir_mhd_ryujones_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ryujones_shock_tube.jl"),
      l2   = [2.34809441e-01, 3.92255943e-01, 8.23575546e-02, 1.75599624e-01, 9.61613519e-01, 6.60825891e-17, 2.15346454e-01, 1.07006529e-01],
      linf = [6.40732148e-01, 9.44889516e-01, 3.54932707e-01, 8.54060243e-01, 2.07757711e+00, 1.11022302e-16, 4.92584725e-01, 2.49526561e-01],
      tspan = (0.0, 0.1))
  end

  @trixi_testset "elixir_mhd_shu_osher_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_shu_osher_shock_tube.jl"),
      l2   = [1.01126210e+00, 8.27157902e+00, 1.30882545e+00, 0.00000000e+00, 5.21930435e+01, 6.56538824e-16, 1.01022340e+00, 0.00000000e+00],
      linf = [2.87172004e+00, 2.26438057e+01, 4.16672442e+00, 0.00000000e+00, 1.35152372e+02, 3.44169138e-15, 2.83556069e+00, 0.00000000e+00],
      tspan = (0.0, 0.2),
      coverage_override = (maxiters=6,))
  end

  @trixi_testset "elixir_mhd_shu_osher_shock_tube.jl with flipped shock direction" begin
    # Include this elixir to make `initial_condition_shu_osher_shock_tube_flipped` available, which is used below
    trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_shu_osher_shock_tube.jl"), tspan=(0.0, 0.0))
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_shu_osher_shock_tube.jl"),
      l2   = [1.01539817e+00, 8.29625810e+00, 1.29548008e+00, 0.00000000e+00, 5.23565514e+01, 3.18641825e-16, 1.00485291e+00, 0.00000000e+00],
      linf = [2.92876280e+00, 2.28341581e+01, 4.11643561e+00, 0.00000000e+00, 1.36966213e+02, 1.55431223e-15, 2.80548864e+00, 0.00000000e+00],
      initial_condition = initial_condition_shu_osher_shock_tube_flipped,
      boundary_conditions=BoundaryConditionDirichlet(initial_condition_shu_osher_shock_tube_flipped),
      tspan = (0.0, 0.2),
      coverage_override = (maxiters=6,))
  end
end

end # module
