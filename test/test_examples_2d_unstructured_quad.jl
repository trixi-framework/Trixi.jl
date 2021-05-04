module TestExamples2DUnstructuredQuad

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Unstructured Curve Mesh for Euler" begin
  @testset "elixir_euler_unstructured_quad.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad.jl"),
      l2   = [7.844127595829963e-5, 0.00011715233141491677, 0.00011715235371677694, 0.0002587793057642201],
      linf = [0.007935754054894772, 0.011476103071539345, 0.01147611262431103, 0.025080478753547464])
  end

  @testset "elixir_euler_unstructured_quad_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad_periodic.jl"),
      l2   = [0.00010720127811104554, 0.00012453133183340216, 0.00012453133183106435, 0.0002862523679720807],
      linf = [0.005838648336111252, 0.008981476526964016, 0.008981476526997323, 0.019987154066631874])
  end

  @testset "elixir_euler_unstructured_quad_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad_free_stream.jl"),
      l2   = [3.357132645873228e-14, 2.3002541067700127e-13, 1.3730189315973952e-13, 4.673831509252854e-13],
      linf = [4.066857961504411e-12, 4.9736201268579805e-11, 2.5309226936442997e-11, 5.4358295642487064e-11],
      tspan=(0.0, 0.1))
  end
end

end # module
