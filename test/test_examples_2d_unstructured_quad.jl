module TestExamples2DUnstructuredQuad

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Unstructured Curve Mesh for Euler" begin
  @testset "elixir_euler_unstructured_quad.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad.jl"),
      l2   = [0.0002948925825934343, 0.0002517795581454954, 0.0002517795583556971, 0.0006357553368178441],
      linf = [0.0049633337805587985, 0.005626613768364486, 0.005626613698727745, 0.01007957474197596],
      polydeg = 3, tspan = (0.0, 0.5))
  end

  @testset "elixir_euler_unstructured_quad_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad_periodic.jl"),
      l2   = [0.00010978828464875207, 0.00013010359527356914, 0.00013010359527326057, 0.0002987656724828824],
      linf = [0.00638626102818618, 0.009804042508242183, 0.009804042508253286, 0.02183139311614468])
  end

  @testset "elixir_euler_unstructured_quad_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad_free_stream.jl"),
      l2   = [3.3596013848623337e-14, 2.4397818684917506e-13, 1.4659046683587473e-13, 4.673119220881569e-13],
      linf = [1.024846874031482e-11, 6.85111412046524e-11, 4.405278919428213e-11, 1.418225537008766e-10],
      tspan = (0.0, 0.1))
  end

  @testset "elixir_euler_unstructured_quad_wall_bc.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad_wall_bc.jl"),
      l2   = [0.0401951408663404, 0.042562446022296446, 0.03734280979760256, 0.10058806400957201],
      linf = [0.24502910304890335, 0.298141188019722, 0.29446571031375074, 0.5937151600027155],
      tspan = (0.0, 0.25))
  end

  @testset "elixir_euler_unstructured_quad_basic.jl" begin
    @test_trixi_include(default_example_unstructured(),
      l2   = [0.0007258658867098887, 0.000676268065087451, 0.0006316238024054346, 0.0014729738442086392],
      linf = [0.004476908674416524, 0.0052614635050272085, 0.004926298866533951, 0.018058026023565432],
      tspan = (0.0, 1.0))
  end
end

end # module
