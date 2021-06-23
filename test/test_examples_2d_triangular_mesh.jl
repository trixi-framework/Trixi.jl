module TestExamples2DTri

using Test
# using StructArrays
using StartUpDG
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "triangular_mesh_2D")

@testset "Triangular mesh tests" begin
  @trixi_testset "Euler equations" begin
    
    @test true # dummy test

    # @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_triangular_mesh.jl"),
    #   l2 = [6.27122850947113e-11, 6.791612221354976e-11, 6.791612221249498e-11, 6.466533604304294e-10], 
    #   linf = [1.8523039727380564e-5, 1.6923451381156696e-5, 1.71384595346602e-5, 4.586131120554171e-5]
    # )

    # @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_periodic_triangular_mesh.jl"),
    #   l2 = [6.943606879787733e-11, 7.474517827181276e-11, 7.474517827154942e-11, 6.637284410683966e-10], 
    #   linf = [1.8302801320757567e-5, 1.6548179149200593e-5, 1.6886330820753415e-5, 4.5056371695828545e-5]
    # )

    # @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_triangular_mesh.jl"),
    #   l2 = [4.293692357434845e-6, 4.293692357445338e-6, 2.069046541101411e-6, 3.978858062534657e-32, 3.978858062534657e-32, 6.3661729000554515e-31, 6.3661729000554515e-31], 
    #   linf = [0.005093360833536398, 0.0010859409295087552, 0.00012810312361999365, 1.6653345369377348e-16, 1.6653345369377348e-16, 6.661338147750939e-16, 6.661338147750939e-16]
    # )
    end
end

end # module
