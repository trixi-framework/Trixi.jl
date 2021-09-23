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
      l2 = [0.1645360902098647, 0.20214759219130263, 0.20214759219130518, 0.10885110164572025, 0.7775556580084889, 0.13112440179705706, 0.13112440179707255, 0.11683502985915632, 0.07605291492939159],
      linf = [0.16355522541989664, 0.2411953506764944, 0.24119535067649692, 0.12198113357293117, 0.7207594672453972, 0.12203245022836606, 0.12203245022842657, 0.10654055347106428, 0.0786832997032684]
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Quad, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, approximation_type=SBP(),
      l2 = [0.15839309853872566, 0.20037359518825335, 0.1973470783087204, 0.10512472553399975, 0.7316831391095003, 0.07405170562480139, 0.07406750411700902, 0.10820988089935167, 0.005259685655454474],
      linf = [0.18394344381280048, 0.2658794192177305, 0.2681905825467176, 0.12609280103892515, 0.784052359590246, 0.06940583878111584, 0.06845019362206828, 0.10946506675354106, 0.009116496688866065],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Tri)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, element_type=Tri(), cfl = 0.5,
      l2 = [0.13510619731411963, 0.15881125803977902, 0.1588112580397801, 0.08579123341962941, 0.6035607100115074, 0.06250113524540919, 0.0625011352454164, 0.0895463314601422, 0.0024591819831242083],
      linf = [0.16218918471176713, 0.2253696302372436, 0.2253696302372458, 0.09410266511145898, 0.6911212423583764, 0.05431413323108791, 0.05431413323109968, 0.08862266720002909, 0.0019260081021027662],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Tri, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, element_type=Tri(), approximation_type = SBP(), cfl = 0.5,
      l2 = [0.15125596853593842, 0.18886909322357132, 0.17893964006102706, 0.09611247221202199, 0.7064983803262789, 0.07328074398250609, 0.07330967419426393, 0.10344172986929333, 0.004993015992204203],
      linf = [0.19907303068030036, 0.26552623057649094, 0.2664190858450435, 0.10905854216866703, 0.7724977235113317, 0.05951091658944896, 0.05897938297692784, 0.09400637935268852, 0.007143885708149413],
    )
  end

end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
