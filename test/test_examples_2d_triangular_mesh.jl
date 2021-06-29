module TestExamples2DTri

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "triangular_mesh_2D")

@testset "Triangular mesh tests" begin    

  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_triangular_mesh.jl"),
    l2 = [7.687514677136661e-5, 8.844681835345058e-5, 8.844681835367038e-5, 0.0002667915678724591], 
    linf = [0.00023033713406972467, 0.0001967281286732181, 0.00019672812868742895, 0.0004570579510136952]
  )

  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_periodic_triangular_mesh.jl"),
    l2 = [8.367502030645914e-5, 9.408647487246263e-5, 9.408647487253276e-5, 0.0002719897210980286], 
    linf = [0.00023033745877265588, 0.000196525001662895, 0.00019652500166067455, 0.000456252647166977]
  )

  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_quadrilateral_mesh.jl"),
    l2 = [1.679658191261153e-5, 1.975052944338864e-5, 1.9750529443605503e-5, 6.243039564033164e-5], 
    linf = [4.21194288451332e-5, 4.386421462854173e-5, 4.386421466406887e-5, 9.206169546338572e-5]
  )
    
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_sbp_triangular_mesh.jl"),
    l2 = [0.025428732930900345, 0.02542873293089407, 0.01783409907430894, 0.0, 0.0, 0.0, 0.0], 
    linf = [0.04266081414305489, 0.04266081414679457, 0.03468583245492329, 0.0, 2.3455572625e-314, 2.3455070497e-314, 0.0]
  )

  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_Triangulate_mesh.jl"),
    l2 = [0.0001016711916046445, 0.00011071422293274785, 0.00011212482087451142, 0.00035893791736543447], 
    linf = [0.00035073781634786805, 0.00039815763002271076, 0.00041642100745109545, 0.0009481311054404529]
  )

end

end # module
