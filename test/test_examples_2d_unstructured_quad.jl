module TestExamples2DUnstructuredQuad

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Unstructured Curve Mesh for Euler" begin
  @testset "elixir_euler_unstructured_quad.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad.jl"),
      l2   = [2.2350052041647245e-5, 3.391208446790789e-5, 3.3911984177238527e-5, 7.652531972096782e-5],
      linf = [0.005276091776615255, 0.007635725818113892, 0.007635614651618772, 0.016749304737659898],
      tspan = (0.0, 0.75))
  end

  @testset "elixir_euler_unstructured_quad_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad_periodic.jl"),
      l2   = [0.00010978828464875207, 0.00013010359527356914, 0.00013010359527326057, 0.0002987656724828824],
      linf = [0.00638626102818618, 0.009804042508242183, 0.009804042508253286, 0.02183139311614468])
  end

  @testset "elixir_euler_unstructured_quad_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad_free_stream.jl"),
      l2   = [3.357132645873228e-14, 2.3002541067700127e-13, 1.3730189315973952e-13, 4.673831509252854e-13],
      linf = [4.066857961504411e-12, 4.9736201268579805e-11, 2.5309226936442997e-11, 5.4358295642487064e-11],
      tspan = (0.0, 0.1))
  end
end

end # module
