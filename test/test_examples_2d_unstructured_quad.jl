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

  @testset "elixir_euler_unstructured_quad.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_unstructured_quad.jl"),
      surface_flux=flux_hll,
      l2   = [2.333716093963336e-5, 3.5123845143806564e-5, 3.51238613975556e-5, 7.878041135955283e-5],
      linf = [0.0027204454304337045, 0.004060183970721054, 0.004060434529746582, 0.009192791921394328])
  end
end

end # module
