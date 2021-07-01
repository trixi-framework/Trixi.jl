module TestExamples2DTri

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "simplicial_3d_dg")

@trixi_testset "3D simplicial mesh tests" begin    

  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_tet_mesh.jl"),
    l2 = [0.0010029534292051608, 0.0011682205957721673, 0.001072975385793516, 0.000997247778892257, 0.0039364354651358294], 
    linf = [0.003660737033303718, 0.005625620600749226, 0.0030566354814669516, 0.0041580358824311325, 0.019326660236036464]
  )  
  
end

end # module
