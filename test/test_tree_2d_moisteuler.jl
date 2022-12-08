module TestExamples2DEuler

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "Moist Compressible Euler" begin

  @trixi_testset "elixir_moist_euler_source_terms_moist.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_moist_euler_source_terms_moist.jl"),
      l2   = [0.001409425605245773, 0.8134961859966655, 0.8144925487175673, 9.015706013660902,
              2.135557653721013e-8, 2.9981381416971306e-8],
      linf = [0.008476629048169926, 1.957595836140905, 1.9590561048679498, 54.613195043231826,
              1.2248000242655582e-7, 1.7375308647843322e-7])
  end
end

end # module
