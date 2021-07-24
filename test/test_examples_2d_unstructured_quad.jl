module TestExamples2DUnstructuredQuad

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "unstructured_2d_dgsem")

@testset "Unstructured Curve Mesh for Euler" begin

  @trixi_testset "elixir_euler_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_periodic.jl"),
      l2   = [0.00010978828464875207, 0.00013010359527356914, 0.00013010359527326057, 0.0002987656724828824],
      linf = [0.00638626102818618, 0.009804042508242183, 0.009804042508253286, 0.02183139311614468])
  end

  @trixi_testset "elixir_euler_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
      l2   = [3.357431396258123e-14, 2.439943089578555e-13, 1.4655386790023588e-13, 4.670410488845425e-13],
      linf = [1.0169198816356584e-11, 6.838458965763294e-11, 4.400946274074613e-11, 1.4071055431941204e-10],
      tspan = (0.0, 0.1),
      atol = 3.0e-13)
  end

  @trixi_testset "elixir_euler_wall_bc.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_wall_bc.jl"),
      l2   = [0.0401951408663404, 0.042562446022296446, 0.03734280979760256, 0.10058806400957201],
      linf = [0.24502910304890335, 0.298141188019722, 0.29446571031375074, 0.5937151600027155],
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
      l2   = [0.06418288595515989, 0.12085170757294837, 0.1208509346385774, 0.07743001850712294,
              0.16221988122575093, 0.04044445575599098, 0.04044451621613186, 0.05735903066057311,
              0.0020955497162158053],
      linf = [0.14169585310192767, 0.3210434288605366, 0.335035261513153, 0.22499513309668773,
              0.44231595436046245, 0.16750863202559962, 0.16753566302125866, 0.17700993590448255,
              0.026783792841174006],
      tspan = (0.0, 0.5))
  end

  @trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [5.376431895262096e-5, 0.09999999205016877, 0.09999999205016812, 0.14142135386740404,
              8.767116802158719e-6, 0.09999999259645813, 0.09999999259645805, 0.14142135397626496,
              1.1559626795792775e-5],
      linf = [0.00039380173292324905, 0.1414487954783921, 0.141448795478425, 0.20003306637526067,
              7.021503823256836e-5, 0.14146450833995794, 0.14146450833994983, 0.20006708807562595,
              0.00013758064589978612],
      tspan = (0.0, 0.5))
  end
end

end # module
