
using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi.jl/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "Linearized Euler Equations 2D" begin
  @trixi_testset "elixir_linearizedeuler_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_linearizedeuler_convergence.jl"),
      l2 = [0.00020601485381444888, 0.00013380483421751216, 0.0001338048342174503, 0.00020601485381444888],
      linf = [0.0011006084408365924, 0.0005788678074691855, 0.0005788678074701847, 0.0011006084408365924]
    )
  end
end
