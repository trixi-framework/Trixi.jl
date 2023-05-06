module TestExamples1DEulerGravity

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi.jl/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_1d_dgsem")

@testset "Compressible Euler with self-gravity" begin
  @trixi_testset "elixir_eulergravity_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
      l2   = [0.0002170799126638106, 0.0002913792848717502, 0.0006112320856262327],
      linf = [0.0004977401033188222, 0.0013594223337776157, 0.002041891084400227])
  end
end

end # module
