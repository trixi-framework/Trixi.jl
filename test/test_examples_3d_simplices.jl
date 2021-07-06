module TestExamples2DTri

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "simplicial_3d_dg")

@testset "3D simplicial mesh tests" begin
  @trixi_testset "elixir_euler_tet_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_tet_mesh.jl"),
      l2 = [0.0010029534292051608, 0.0011682205957721673, 0.001072975385793516, 0.000997247778892257, 0.0039364354651358294],
      linf = [0.003660737033303718, 0.005625620600749226, 0.0030566354814669516, 0.0041580358824311325, 0.019326660236036464]
    )
  end
end

@testset "3D simplicial flux differencing tests" begin

  @trixi_testset "elixir_euler_tet_mesh_flux_diff.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_tet_mesh_flux_diff.jl"),
      l2 = [0.04693516732847953, 0.0490148086446717, 0.048587225155397755, 0.04921946695784349, 0.12147656172990916],
      linf = [0.10277064447821016, 0.12549628202719165, 0.11540362612261723, 0.12344556494598535, 0.3288299122603555]
    )
  end

end

end # module
