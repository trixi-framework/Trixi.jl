module TestExamples2DEulerAcoustics

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")

@testset "Acoustic perturbation coupled with compressible Euler" begin
  @trixi_testset "elixir_euleracoustics_co-rotating_vortex_pair.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euleracoustics_co-rotating_vortex_pair.jl"),
                        initial_refinement_level=5,
                        tspan1=(0.0, 1.0), tspan_averaging=(0.0, 1.0), tspan=(0.0, 1.0),
      l2 = [0.0001320604098963066, 0.00013176913587282997, 0.0002160812547568378, 13.000001753052986, 26.000000802430048, 38.00000884739542, 51.000000003859434],
      linf = [0.2273798192285465, 0.15267816065649403, 0.25077014613342513, 13.467874737415409, 26.546588241879213, 38.13775493609699, 51.38017788683809]
    )
  end
end

end # module