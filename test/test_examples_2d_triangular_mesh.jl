module TestExamples2DTri

using Test
using StructArrays
using StartUpDG
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "triangular_mesh_2D")

@testset "Triangular mesh tests" begin
  @trixi_testset "Euler equations" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_triangular_mesh.jl"),
      l2   = [1.495997057472888e-10, 1.5857851470266195e-10, 1.5857851469785578e-10, 9.81342164163901e-10],
      linf = [2.4642315644918256e-5, 2.108887656504521e-5, 2.910954871415683e-5, 7.78236537710697e-5])

    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_periodic_triangular_mesh.jl"),
      l2   = [1.5993989713310647e-10, 1.7121787580224833e-10, 1.712178758032254e-10, 9.941518664579888e-10],
      linf = [2.3819208376352208e-5, 2.6706116870389707e-5, 2.0829004917732874e-5, 5.768496226554731e-5])
    end
end

end # module
