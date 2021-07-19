module TestExamples2DAPEEuler

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "Acoustic Perturbation coupled with compressible Euler" begin
  @trixi_testset "elixir_ape_euler_co-rotating_vortex_pair.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_euler_co-rotating_vortex_pair.jl"),
                        initial_refinement_level=5,
                        tspan1=(0.0, 1.0), tspan_averaging=(0.0, 1.0), tspan=(0.0, 1.0),
      l2   = [0.00013510045122922684, 0.0001333171869560621, 0.00019539947240857303, 13.000001759776172, 26.00000080550468, 37.99808866866672, 50.998079808126604],
      linf = [0.23847116721559336, 0.16011856602324562, 0.20030228830185642, 13.468670035749438, 26.547578962378058, 38.13606473287717, 51.37900693683955]
    )
  end
end


end # module