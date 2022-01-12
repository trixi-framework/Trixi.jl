module TestExamplesKPP

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "KPP" begin
  @trixi_testset "elixir_kpp.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_kpp.jl"),
      l2   = [5.17986197e-01],
      linf = [8.93080899e+00],
      tspan = (0.0, 0.1),
      atol = 1e-6,
      rtol = 1e-6)
  end
end

end # module
