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
      l2 = [0.1564827900943962, 0.1966824470701965, 0.19668244707019755, 0.10639504495017124, 0.7205880727756032, 0.07644160038961999, 0.07644160038962199, 0.11138522669651309, 0.0023303802856204],
      linf = [0.16567998381188986, 0.24548085844852077, 0.2454808584485253, 0.1195052326533841, 0.7434508609863526, 0.06260342757770387, 0.06260342757770587, 0.10589713961056058, 0.002174385718228176],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Quad, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, approximation_type=SBP(),
      l2 = [0.15848410002354318, 0.20025149896737277, 0.1972351792975989, 0.10509670983097537, 0.7318965818592882, 0.07395758572410727, 0.07397453931747068, 0.10809218599510823, 0.005038271220315641],
      linf = [0.18452601698719973, 0.2657462069235125, 0.26811127976910887, 0.1261124250308698, 0.7834984064324986, 0.06933409029034898, 0.06850219018813974, 0.10922544551648639, 0.008733639714635356],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Tri)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, element_type=Tri(), cfl = 2.0,
      l2 = [0.13510338804353805, 0.15877636976575796, 0.1587763697657592, 0.08578259205071914, 0.6035258981437235, 0.062488056965370756, 0.06248805696537788, 0.08950239564181779, 0.0023849354348949186],
      linf = [0.16203206344758803, 0.22521343578645228, 0.22521343578645484, 0.09418067220223929, 0.690645551665038, 0.05440013855684356, 0.05440013855685577, 0.08842206301431799, 0.0019062562942840076],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Tri, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, element_type=Tri(), approximation_type = SBP(), cfl = 2.0,
      l2 = [0.1513010983665277, 0.18881599408794006, 0.17886092635725412, 0.09608793499496418, 0.7063121520186653, 0.07319555768044278, 0.07321495476588985, 0.10335465144717838, 0.004712077339914576],
      linf = [0.19927590199897482, 0.2654144312283347, 0.2668958452901072, 0.10897505729625148, 0.7730746954119496, 0.059240626999394586, 0.05867808018652387, 0.09391247121508028, 0.006854677027935561],
    )
  end

end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
