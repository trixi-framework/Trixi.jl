module TestExamplesDGMulti2D

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "dgmulti_2d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "DGMulti 2D" begin
  @trixi_testset "elixir_euler_weakform.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      l2 = [0.0013463150267220328, 0.0014235793662296975, 0.0014235793662300024, 0.00472191786388071],
      linf = [0.0015248303482329195, 0.0020707330926952316, 0.002070733092696342, 0.004913455679613321],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      approximation_type = SBP(),
      l2 = [0.007465016439747669, 0.005297423547392931, 0.005297423547403158, 0.01470161132498598],
      linf = [0.021489935389522374, 0.013528869419211276, 0.013528869418737433, 0.03269072432621112],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (Quadrilateral elements)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      element_type = Quad(),
      l2 = [0.000290967383489527, 0.00028113809346926776, 0.0002811380934695505, 0.00102004771420306],
      linf = [0.0004970344840584673, 0.00040590009306518127, 0.0004059000930629608, 0.0014247732095258314],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (EC) " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      l2 = [0.008110785816182155, 0.0074686552093368745, 0.007468655209336097, 0.015986513837074563],
      linf = [0.01230954687917274, 0.013884805356942254, 0.013884805356973784, 0.040387377818142056],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      approximation_type = SBP(),
      l2 = [0.01285864081726596, 0.010650165503847099, 0.01065016550381281, 0.026286162111579015],
      linf = [0.037333313274372504, 0.05308320130762212, 0.05308320130841948, 0.13378665881805185],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (Quadrilateral elements, SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      element_type = Quad(),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      approximation_type = SBP(),
      l2 = [0.0029319624187308896, 0.0030625695968579886, 0.003062569596855081, 0.006843948320775483],
      linf = [0.013700713240587747, 0.012810144682950497, 0.01281014468295072, 0.022412654330661308],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (convergence)" begin
    mean_convergence = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"), 2)
    @test isapprox(mean_convergence[:l2], [4.249875508800025, 4.133727008051228, 4.133727007601049, 4.086238794189699], rtol=0.05)
  end

  @trixi_testset "elixir_euler_weakform_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
      l2 = [0.0014986508075708323, 0.001528523420746786, 0.0015285234207473158, 0.004846505183839211],
      linf = [0.0015062108658376872, 0.0019373508504645365, 0.0019373508504538783, 0.004742686826709086]
    )
  end

  @trixi_testset "elixir_euler_triangulate_pkg_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_triangulate_pkg_mesh.jl"),
      l2 = [4.664661209491976e-6, 3.7033509525940745e-6, 4.794877426562555e-6, 1.2682723101532175e-5],
      linf = [2.5099852761334418e-5, 2.2683684021362893e-5, 2.6180448559287584e-5, 5.5752932611508044e-5]
    )
  end

  @trixi_testset "elixir_euler_kelvin_helmholtz_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_kelvin_helmholtz_instability.jl"),
      cells_per_dimension = (8,8), tspan = (0.0, 0.2),
      l2 = [0.11140371194266245, 0.06598148257331912, 0.10448950567476646, 0.1602319547232773],
      linf = [0.2403274017090733, 0.1659961476843728, 0.12354683515032569, 0.2691592411995334],
    )
  end

  @trixi_testset "elixir_euler_rayleigh_taylor_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_rayleigh_taylor_instability.jl"),
      cells_per_dimension = (8,8), tspan = (0.0, 0.2),
      l2 = [0.03548329484912729, 0.002591414376082683, 0.006916327792623457, 0.016235069002818153],
      linf = [0.4783963902824797, 0.022527207050681054, 0.040307056293369226, 0.0852365428206836],
    )
  end

  @trixi_testset "elixir_euler_brown_minion_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_brown_minion_vortex.jl"),
      num_cells_per_dimension = 4, tspan = (0.0, 0.1),
      l2 = [0.006818116685388498, 0.02123358215096057, 0.009928432022103199, 0.15364518920518527],
      linf = [0.015838346753841548, 0.09449772987646823, 0.021276761322582504, 0.3470925282340751],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Quad)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4,
      l2 = [0.1564358917715464, 0.19660983101629445, 0.19660983101629576, 0.10643232556383637, 0.72047626377434, 0.07648249832045073, 0.07648249832045254, 0.11144797318021817, 0.002331007575596304],
      linf = [0.16562144860104122, 0.24532680982697921, 0.24532680982698318, 0.11930485187308991, 0.7432867515284758, 0.06264144377305736, 0.06264144377305936, 0.10605481301778641, 0.0021579797455846066],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Quad, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, approximation_type=SBP(),
      l2 = [0.1584008010621207, 0.20036876121002614, 0.1973415606984194, 0.1051232492951558, 0.731666369248373, 0.07402858337450637, 0.07404252233428216, 0.1082114087171412, 0.0050728905105294035],
      linf = [0.18380582927714728, 0.2658922909334918, 0.26826405420346583, 0.12608620166294765, 0.7838850108145161, 0.06945276592898098, 0.06850196760397975, 0.10944602059844832, 0.008626685526959006],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Tri)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, element_type=Tri(), cfl = 2.0,
      l2 = [0.13510933840569686, 0.15880447944871773, 0.15880447944871903, 0.0857910183414984, 0.6035138361490541, 0.062491848873538314, 0.06249184887354542, 0.08954121627204585, 0.002377526761285085],
      linf = [0.16217655493484107, 0.22535223617374303, 0.22535223617374667, 0.09409286605961527, 0.6911025718677486, 0.05441055736464673, 0.054410557364658274, 0.08862136660725306, 0.0018998946311014448],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Tri, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, element_type=Tri(), approximation_type = SBP(), cfl = 2.0,
      l2 = [0.15126178178260977, 0.1888495015832068, 0.17892960814973943, 0.09611064051690098, 0.7062597195315075, 0.07327249227333649, 0.07329173734052341, 0.10342647099474465, 0.004676375009443173],
      linf = [0.19912799309678353, 0.2656085491966023, 0.2663805780580941, 0.10907369472506918, 0.7720757083858598, 0.059259671707533546, 0.05872550253860509, 0.09402068210036163, 0.006864417458036482],
    )
  end

end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
