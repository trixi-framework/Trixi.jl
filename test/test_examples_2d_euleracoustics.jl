module TestExamples2DEulerAcoustics

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "Acoustic perturbation coupled with compressible Euler" begin
  @trixi_testset "elixir_euleracoustics_co-rotating_vortex_pair.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euleracoustics_co-rotating_vortex_pair.jl"),
                        initial_refinement_level=5,
                        tspan1=(0.0, 1.0), tspan_averaging=(0.0, 1.0), tspan=(0.0, 1.0),
      l2 = [0.00013455394678581038, 0.00013349316046130501, 0.00019398895149959965, 13.000001753048673, 26.00000080242852, 38.00000884739933, 51.00000000385884],
      linf = [0.23549521398132314, 0.16043592784039867, 0.1982245258894231, 13.467848922104073, 26.546577577240164, 38.137770176967585, 51.38016286693769]
    )
  end
end

end # module