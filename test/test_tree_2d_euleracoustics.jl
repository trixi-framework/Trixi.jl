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
      l2 = [0.00013268029905807722, 0.0001335062197031223, 0.00021776333678401362, 13.000001753042364, 26.00000080243847, 38.00000884725549, 51.000000003859995],
      linf = [0.22312716933051027, 0.1579924424942319, 0.25194831158255576, 13.468872744263273, 26.54666679978679, 38.139032147739684, 51.378134660241294]
    )
  end
end

end # module