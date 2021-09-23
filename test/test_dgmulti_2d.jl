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
      l2 = [0.0013536930300254945, 0.0014315603442106193, 0.001431560344211359, 0.0047393341007602625],
      linf = [0.001514260921466004, 0.0020623991944839215, 0.002062399194485476, 0.004897700392503701],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      approximation_type = SBP(),
      l2 = [0.0074706882014934735, 0.005306220583603261, 0.005306220583613591, 0.014724842607716771],
      linf = [0.021563604940952885, 0.01359397832530762, 0.013593978324845324, 0.03270995869587523]
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (Quadrilateral elements)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      element_type = Quad(),
      l2 = [0.00031892254415307093, 0.00033637562986771894, 0.0003363756298680649, 0.0011100259064243145],
      linf = [0.001073298211445639, 0.0013568139808282087, 0.0013568139808290969, 0.0032249020004324613]
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (EC) " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      l2 = [0.007801417730672109, 0.00708583561714128, 0.0070858356171393, 0.015217574294198809],
      linf = [0.011572828457858897, 0.013965298735070686, 0.01396529873508534, 0.04227683691807904],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      approximation_type = SBP(),
      l2 = [0.01280067571168776, 0.010607599608273302, 0.010607599608239775, 0.026408338014056548],
      linf = [0.037983023185674814, 0.05321027922533417, 0.05321027922608157, 0.13392025411844033],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (Quadrilateral elements, SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4,4),
      element_type = Quad(),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      approximation_type = SBP(),
      l2 = [0.0029373718090697975, 0.0030629360605489465, 0.003062936060545615, 0.0068486089344859755],
      linf = [0.01360165305316885, 0.01267402847925303, 0.012674028479251254, 0.02210545278615017],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (convergence)" begin
    mean_convergence = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"), 2)
    @test isapprox(mean_convergence[:l2], [4.243843382379403, 4.128314378833922, 4.128314378397532, 4.081366752807379], rtol=0.05)
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
      l2 = [0.15627877590658854, 0.196822294797119, 0.19682229479712005, 0.10642820161010955, 0.7215145028741121, 0.07672764569974329, 0.07672764569974469, 0.11147102976872604, 0.006618676365857434],
      linf = [0.164632039640624, 0.24438577987285662, 0.24438577987285925, 0.11957432337123705, 0.7453885865037737, 0.06312338025257125, 0.06312338025257525, 0.10610163532094796, 0.006611235811239196],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Quad, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, approximation_type=SBP(), cfl = .5,
      l2 = [0.15819852787432703, 0.2005817748602469, 0.19755861113059794, 0.10507976435796945, 0.7334250302876919, 0.07453624613978425, 0.07452738383366823, 0.10807854091049794, 0.01060506973555037],
      linf = [0.1855766240255481, 0.26713755684041474, 0.2687581432173838, 0.1260329552527552, 0.7906407572419436, 0.07054303647451743, 0.06883034551721989, 0.1092119039206263, 0.015959667603072346],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Tri)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, element_type=Tri(),
      l2 = [0.13496937069619225, 0.1588347750795757, 0.15883477507957702, 0.08572446966868995, 0.6032811347234383, 0.06278588915610084, 0.06278588915610814, 0.08950064241166451, 0.0036668958539640916],
      linf = [0.16288247211931828, 0.22569764389960817, 0.22569764389961244, 0.09450242782062232, 0.6912712434561352, 0.05307474545518964, 0.05307474545519808, 0.08859642212442709, 0.0047652592187580525],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Tri, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, element_type=Tri(), approximation_type = SBP(), cfl = .5,
      l2 = [0.15127981853435116, 0.1894746829321331, 0.17971661925546695, 0.09619672299310533, 0.7060076038236979, 0.07352535439944866, 0.07333669235299438, 0.10355591748395487, 0.006993463177228327],
      linf = [0.19878852251592938, 0.2646521511970933, 0.26796931932114965, 0.10908661431838862, 0.7734476220151678, 0.06107876410327462, 0.061988856523786495, 0.09365502259154623, 0.017323196252498743],
    )
  end

end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
