module TestExamplesUnstructuredMesh2D

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "unstructured_2d_dgsem")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "UnstructuredMesh2D" begin
  @trixi_testset "elixir_euler_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_periodic.jl"),
      l2   = [0.00010978828464875207, 0.00013010359527356914, 0.00013010359527326057, 0.0002987656724828824],
      linf = [0.00638626102818618, 0.009804042508242183, 0.009804042508253286, 0.02183139311614468])
  end

  @trixi_testset "elixir_euler_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
      l2   = [3.357431396258123e-14, 2.439943089578555e-13, 1.4655386790023588e-13, 4.670410488845425e-13],
      linf = [1.0169198816356584e-11, 6.838458965763294e-11, 4.456759961080081e-11, 1.4207657272891083e-10],
      tspan = (0.0, 0.1),
      atol = 3.0e-13)
  end

  @trixi_testset "elixir_euler_wall_bc.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_wall_bc.jl"),
      l2   = [0.040189107976346644, 0.04256154998030852, 0.03734120743842209, 0.10057425897733507],
      linf = [0.24455374304626365, 0.2970686406973577, 0.29339040847600434, 0.5915610037764794],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_euler_basic.jl" begin
    @test_trixi_include(default_example_unstructured(),
      l2   = [0.0007213418215265047, 0.0006752337675043779, 0.0006437485997536973, 0.0014782883071363362],
      linf = [0.004301288971032324, 0.005243995459478956, 0.004685630332338153, 0.01750217718347713],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_euler_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_restart.jl"),
      l2   = [0.0007213418215265047, 0.0006752337675043779, 0.0006437485997536973, 0.0014782883071363362],
      linf = [0.004301288971032324, 0.005243995459478956, 0.004685630332338153, 0.01750217718347713])
  end

  @trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.06594600495903137, 0.10803914821786433, 0.10805946357846291, 0.1738171782368222],
      linf = [0.31880214280781305, 0.3468488554333352, 0.34592958184413264, 0.784555926860546],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [0.00018729339078205488], 
      linf = [0.0018997287705734278])
  end

  @trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
      l2   = [2.19945600e-01, 1.71050453e-01, 1.71050453e-01, 1.21719195e+00],
      linf = [7.44218635e-01, 7.02887039e-01, 7.02887039e-01, 6.11732719e+00],
      tspan = (0.0, 0.3))
  end

  @trixi_testset "elixir_acoustics_gauss_wall.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_acoustics_gauss_wall.jl"),
      l2   = [0.029330394861252995, 0.029345079728907965, 0.03803795043486467, 0.0,
              7.175152371650832e-16, 1.4350304743301665e-15, 1.4350304743301665e-15],
      linf = [0.36236334472179443, 0.3690785638275256, 0.8475748723784078, 0.0,
              8.881784197001252e-16, 1.7763568394002505e-15, 1.7763568394002505e-15],
      tspan = (0.0, 5.0))
  end

  @trixi_testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [0.06418293357851637, 0.12085176618704108, 0.12085099342419513, 0.07743005602933221,
              0.1622218916638482, 0.04044434425257972, 0.04044440614962498, 0.05735896706356321,
              0.0020992340041681734],
      linf = [0.1417000509328017, 0.3210578460652491, 0.335041095545175, 0.22500796423572675,
              0.44230628074326406, 0.16743171716317784, 0.16745989278866702, 0.17700588224362557,
              0.02692320090677309],
      tspan = (0.0, 0.5))
  end

  @trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [5.377518922553881e-5, 0.09999999206243514, 0.09999999206243441, 0.1414213538550799,
              8.770450430886394e-6, 0.0999999926130084, 0.0999999926130088, 0.14142135396487032,
              1.1553833987291942e-5],
      linf = [0.00039334982566352483, 0.14144904937275282, 0.14144904937277897, 0.20003315928443416,
              6.826863293230012e-5, 0.14146512909995967, 0.14146512909994702, 0.20006706837452526,
              0.00013645610312810813],
      tspan = (0.0, 0.5))
  end

  @trixi_testset "elixir_shallowwater_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_ec.jl"),
      l2   = [0.6106939484178353, 0.48586236867426724, 0.48234490854514356, 0.29467422718511727],
      linf = [2.775979948281604, 3.1721242154451548, 3.5713448319601393, 2.052861364219655],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
      l2   = [1.2164292510839083, 2.8049631232564753e-12, 1.5664980749498454e-12, 1.216429251083908],
      linf = [1.5138512282315868, 5.0263880749562876e-11, 3.5805418805003075e-11, 1.513851228231574],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      l2   = [0.0011197623982310795, 0.04456344888447023, 0.014317376629669337, 5.089218476758975e-6],
      linf = [0.007835284004819698, 0.3486891284278597, 0.11242778979399048, 2.6407324614119432e-5],
      tspan = (0.0, 0.025))
  end

  @trixi_testset "elixir_shallowwater_dirichlet.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_dirichlet.jl"),
      l2   = [1.1577518608963063e-5, 4.371173779970303e-13, 4.2152984234036224e-13, 1.1577518608935235e-5],
      linf = [8.394063878491842e-5, 9.632644747754261e-11, 9.54898901893391e-11, 8.394063879602065e-5],
      tspan = (0.0, 2.0))
  end

  @trixi_testset "elixir_shallowwater_wall_bc.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_wall_bc.jl"),
      l2   = [0.04443210036005491, 0.14454582374113853, 0.15239799057671485, 6.225080477024867e-8],
      linf = [0.7727399447958347, 2.127376144492187, 3.361677723990531, 3.982097160903919e-7],
      tspan = (0.0, 0.05))
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # module
