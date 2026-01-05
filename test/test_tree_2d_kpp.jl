module TestExamplesKPP

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")

@testset "KPP" begin
#! format: noindent

@trixi_testset "elixir_kpp.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_kpp.jl"),
                        l2=[0.36563290910786106],
                        linf=[9.116732052340398],
                        max_refinement_level=6,
                        tspan=(0.0, 0.01),
                        atol=1e-6,
                        rtol=1e-6,)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end

end # module
