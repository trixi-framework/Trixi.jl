module TestTree2DFDSBP

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_fdsbp")

@testset "Linear scalar advection" begin
  @trixi_testset "elixir_advection_extended.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [2.899428330179038e-6], 
      linf = [8.433122756446032e-6],
      rtol = 5.0e-7) # These results change a little bit and depend on the CI system
  end
end

end # module
