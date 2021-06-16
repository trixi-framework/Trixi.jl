module TestExamples2DUnstructuredQuad

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Unstructured Curve Mesh for Euler" begin

  @trixi_testset "elixir_euler_unstructured_quad_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad_periodic.jl"),
      l2   = [0.00010978828464875207, 0.00013010359527356914, 0.00013010359527326057, 0.0002987656724828824],
      linf = [0.00638626102818618, 0.009804042508242183, 0.009804042508253286, 0.02183139311614468])
  end

  @trixi_testset "elixir_euler_unstructured_quad_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad_free_stream.jl"),
      l2   = [3.3596013848623337e-14, 2.4397818684917506e-13, 1.4659046683587473e-13, 4.673119220881569e-13],
      linf = [1.024846874031482e-11, 6.85111412046524e-11, 4.405278919428213e-11, 1.418225537008766e-10],
      tspan = (0.0, 0.1))
  end

  @trixi_testset "elixir_euler_unstructured_quad_wall_bc.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad_wall_bc.jl"),
      l2   = [0.0401951408663404, 0.042562446022296446, 0.03734280979760256, 0.10058806400957201],
      linf = [0.24502910304890335, 0.298141188019722, 0.29446571031375074, 0.5937151600027155],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_euler_unstructured_quad_basic.jl" begin
    @test_trixi_include(default_example_unstructured(),
      l2   = [0.0007258658867098887, 0.000676268065087451, 0.0006316238024054346, 0.0014729738442086392],
      linf = [0.004476908674416524, 0.0052614635050272085, 0.004926298866533951, 0.018058026023565432],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_euler_unstructured_quad_basic_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad_basic_restart.jl"),
      l2   = [0.0007258658867098887, 0.000676268065087451, 0.0006316238024054346, 0.0014729738442086392],
      linf = [0.004476908674416524, 0.0052614635050272085, 0.004926298866533951, 0.018058026023565432])
  end

  @trixi_testset "elixir_euler_unstructured_quad_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad_ec.jl"),
      l2   = [0.06594600495903137, 0.10803914821786433, 0.10805946357846291, 0.1738171782368222],
      linf = [0.31880214280781305, 0.3468488554333352, 0.34592958184413264, 0.784555926860546],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_advection_unstructured_quad_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_unstructured_quad_basic.jl"),
      l2   = [0.00023219572238346008],
      linf = [0.0017401556568237275])
  end

  @trixi_testset "elixir_ape_unstructured_quad_gauss_wall.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_unstructured_quad_gauss_wall.jl"),
      l2   = [0.029331247985489625, 0.02934616721732521, 0.03803253571320854, 0.0,
              7.465932985352019e-16, 1.4931865970704038e-15, 1.4931865970704038e-15],
      linf = [0.3626825396196784, 0.3684490307932018, 0.8477478712580901, 0.0,
              8.881784197001252e-16, 1.7763568394002505e-15, 1.7763568394002505e-15],
      tspan = (0.0, 5.0))
  end

  @trixi_testset "elixir_mhd_unstructured_quad_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_unstructured_quad_ec.jl"),
      l2   = [0.06410416843183596, 0.12088701564851412, 0.12088660282556184, 0.07736936335775574,
              0.16258858668184714, 0.0404471651213124, 0.04044684898917552, 0.05741636416434423,
              0.0021379094385133315],
      linf = [0.13505810638721094, 0.3164618450494714, 0.3226127662385151, 0.2050483839890379,
              0.43905760760113655, 0.16862877472930893, 0.16878393786562973, 0.197259170172277,
              0.027529614504985696],
      tspan = (0.0, 0.5))
  end

  @trixi_testset "elixir_mhd_unstructured_quad_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_unstructured_quad_alfven_wave.jl"),
      l2   = [5.377518922553881e-5, 0.09999999206243514, 0.09999999206243441, 0.1414213538550799,
              8.770450430886394e-6, 0.0999999926130084, 0.0999999926130088, 0.14142135396487032,
              1.1553833987291942e-5],
      linf = [0.00039334982566352483, 0.14144904937275282, 0.14144904937277897, 0.20003315928443416,
              6.826863293230012e-5, 0.14146512909995967, 0.14146512909994702, 0.20006706837452526,
              0.00013645610312810813],
      tspan = (0.0, 0.5))
  end
end

end # module
