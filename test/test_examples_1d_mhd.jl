module TestExamples1DMHD

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
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
      l2   = [1.4717982305884098e-5, 1.182263100132273e-5, 2.3571203951257584e-5, 2.3571203951305505e-5, 1.8950292712340346e-6, 1.196322416573199e-16, 2.3661336207200768e-5, 2.366133620713721e-5],
      linf = [6.156423568670633e-5, 3.474803933150424e-5, 0.00011882557236665703, 0.0001188255723667403, 6.861887285491974e-6, 2.220446049250313e-16, 0.00012195115526943134, 0.00012195115526922318])
  end

  @trixi_testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [0.15286156171121895, 0.16926639610967018, 0.30495838128766584, 0.0, 0.345735219586204, 6.608258910237594e-17, 0.31766258436717354, 0.0],
      linf = [0.7012110703585162, 0.508655909370987, 1.251766990108334, 0.0, 1.9957985139014158, 1.1102230246251565e-16, 1.6877793846395974, 0.0])
  end

  @trixi_testset "elixir_mhd_briowu_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_briowu_shock_tube.jl"),
      l2   = [0.17764301067932906, 0.19693621875378622, 0.3635136528288642, 0.0, 0.3757321708837591, 8.593007507325741e-16, 0.36473438378159656, 0.0],
      linf = [0.5601530250396535, 0.43867368105486537, 1.0960903616351099, 0.0, 1.0551794137886303, 4.107825191113079e-15, 1.5374410890043144, 0.0])
  end

  @trixi_testset "elixir_mhd_torrilhon_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_torrilhon_shock_tube.jl"),
      l2   = [0.45702524713047815, 0.4792453322653316, 0.3406561762729464, 0.4478037841848423, 0.9204646319222438, 1.3216517820475193e-16, 0.2889772516628848, 0.2552071770333965],
      linf = [1.221507578527215, 0.8927062606782392, 0.8536666880170606, 0.9739151339925485, 1.6602343391033632, 2.220446049250313e-16, 0.686743631371773, 0.6555428379163893])
  end

  @trixi_testset "elixir_mhd_ryujones_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ryujones_shock_tube.jl"),
      l2   = [0.23480132779180388, 0.3922573469657149, 0.08236145370312341, 0.17559850888696307, 0.9615951543069574, 6.608258910237605e-17, 0.21534816808200136, 0.10699642480726586],
      linf = [0.6397374584516875, 0.9457640306380323, 0.3546254702548819, 0.8532865350683194, 2.078210620993694, 1.1102230246251565e-16, 0.49334335395420337, 0.2504525070653655],
      tspan = (0.0, 0.1))
  end

  @trixi_testset "elixir_mhd_shu_osher_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_shu_osher_shock_tube.jl"),
      l2   = [1.011173124884092, 8.271206982561576, 1.3087914055352385, 0.0, 52.19213714506333, 6.743357635607637e-16, 1.0102652381088522, 0.0],
      linf = [2.8649230108993886, 22.600699637929466, 4.166209106166631, 0.0, 135.11479022821882, 3.4416913763379853e-15, 2.83396622829752, 0.0],
      tspan = (0.0, 0.2))
  end

  @trixi_testset "elixir_mhd_shu_osher_shock_tube.jl with flipped shock direction" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_shu_osher_shock_tube.jl"),
      l2   = [1.0150653343383023, 8.293796006442639, 1.2953342215936525, 0.0, 52.34290101591115, 3.352946970627566e-16, 1.0046607148062305, 0.0],
      linf = [2.9091039190988806, 22.734543670960285, 4.124890055211936, 0.0, 136.89057030285923, 1.5543122344752192e-15, 2.806658251289774, 0.0],
      initial_condition = initial_condition_shu_osher_shock_tube_flipped,
      boundary_conditions=BoundaryConditionDirichlet(initial_condition_shu_osher_shock_tube_flipped),
      tspan = (0.0, 0.2))
  end
end

end # module
