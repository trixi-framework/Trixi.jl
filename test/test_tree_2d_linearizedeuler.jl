module TestExamples2DEulerLinearized

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")

@testset "Linearized Euler Equations 2D" begin
  @trixi_testset "elixir_linearized_euler_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_linearized_euler_convergence_test.jl"),
                        initial_refinement_level=4,
                        tspan1=(0.0, 0.2), tspan_averaging=(0.0, 0.2), tspan=(0.0, 0.2),
                        l2 = [0.00020601485381444888, 0.00013380483421751216, 0.0001338048342174503, 0.00020601485381444888],
                        linf = [0.0011006084408365924, 0.0005788678074691855, 0.0005788678074701847, 0.0011006084408365924]
    )
  end
end

end # module