module TestExamples2DTri

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "simplicial_3d_dg")

@testset "3D simplicial mesh" begin
  @trixi_testset "elixir_euler_tet_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_tet_mesh.jl"),
      l2 = [0.0010029534292051608, 0.0011682205957721673, 0.001072975385793516, 0.000997247778892257, 0.0039364354651358294],
      linf = [0.003660737033303718, 0.005625620600749226, 0.0030566354814669516, 0.0041580358824311325, 0.019326660236036464]
    )
  end

  @trixi_testset "elixir_euler_tet_mesh_flux_diff.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_tet_mesh_flux_diff.jl"),
      l2 = [0.10199177415993853, 0.11613427360212616, 0.11026702646784316, 0.10994524014778534, 0.2582449717237713],
      linf = [0.2671818932369674, 0.2936313498471166, 0.34636540613352573, 0.2996048158961957, 0.7082318124804874]
    )
  end

end

end # module
