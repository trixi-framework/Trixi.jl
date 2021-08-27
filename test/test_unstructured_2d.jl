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
      l2   = [0.040188664079620025, 0.04256486972540654, 0.037340214359279124, 0.10057462643345545],
      linf = [0.24456637650533386, 0.29664875461338475, 0.2913090588628896, 0.5915655698655942],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_euler_basic.jl" begin
    @test_trixi_include(default_example_unstructured(),
      l2   = [0.0007258658867098887, 0.000676268065087451, 0.0006316238024054346, 0.0014729738442086392],
      linf = [0.004476908674416524, 0.0052614635050272085, 0.004926298866533951, 0.018058026023565432],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_euler_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_restart.jl"),
      l2   = [0.0007258658867098887, 0.000676268065087451, 0.0006316238024054346, 0.0014729738442086392],
      linf = [0.004476908674416524, 0.0052614635050272085, 0.004926298866533951, 0.018058026023565432])
  end

  @trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.06594600495903137, 0.10803914821786433, 0.10805946357846291, 0.1738171782368222],
      linf = [0.31880214280781305, 0.3468488554333352, 0.34592958184413264, 0.784555926860546],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [0.00023219572238346008],
      linf = [0.0017401556568237275])
  end

  @trixi_testset "elixir_ape_gauss_wall.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_gauss_wall.jl"),
      l2   = [0.029331247985489625, 0.02934616721732521, 0.03803253571320854, 0.0,
              7.465932985352019e-16, 1.4931865970704038e-15, 1.4931865970704038e-15],
      linf = [0.3626825396196784, 0.3684490307932018, 0.8477478712580901, 0.0,
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
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # module
