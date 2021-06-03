module TestTree2DFDSBP

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_fdsbp")

@testset "Linear scalar advection" begin
  @testset "elixir_advection_extended.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [2.29497319e-06],
      linf = [8.48027655e-06])
  end
end

end # module
