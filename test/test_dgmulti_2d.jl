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
      l2 = [0.1563296077431528, 0.19689169168905057, 0.19689169168905118, 0.10639159494970754, 0.7216222011801184, 0.07668258658081525, 0.0766825865808167, 0.11141071407243605, 0.0066214056295862515],
      linf = [0.16471878624987513, 0.24457559000083715, 0.24457559000083842, 0.119771976909315, 0.7455734331018422, 0.06308166727440323, 0.0630816672744059, 0.10596797104423084, 0.006612894541423419],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Quad, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, approximation_type=SBP(),
      l2 = [0.1583034913753957, 0.20050572416564735, 0.19750386982921114, 0.10504630898170826, 0.733748151155864, 0.07446477114887506, 0.07446024796990044, 0.10795868986518672, 0.010662543792446768],
      linf = [0.1861580507519166, 0.26716143204981085, 0.2687788142276168, 0.1260782458811922, 0.7905539980653713, 0.07054490008868419, 0.06886627410310897, 0.10903347271427677, 0.01604106855436705],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Tri)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, element_type=Tri(),
      l2 = [0.1349654355772558, 0.1588038334891554, 0.1588038334891567, 0.08571945890876945, 0.6032920444317149, 0.06278143421347503, 0.06278143421348256, 0.08946842287263297, 0.0036506137157130527],
      linf = [0.16274630801574053, 0.2255567378426629, 0.22555673784266667, 0.09457200508819374, 0.6908520815816002, 0.053067340166437216, 0.053067340166445875, 0.08841222728946285, 0.00474295833506402],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Tri, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      num_cells_per_dimension = 4, element_type=Tri(), approximation_type = SBP(),
      l2 = [0.15136191155700096, 0.18946751746009416, 0.1796867743058775, 0.09625792194830064, 0.7062241049681947, 0.07346282087079924, 0.07337240729938442, 0.10360528367980766, 0.007021581284842776],
      linf = [0.19878323450625868, 0.2645511415480556, 0.26821985742204774, 0.10904281386998915, 0.7743265951127776, 0.06075518436421179, 0.061857839803176384, 0.09397814084616862, 0.0176234829760554],
    )
  end

end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
