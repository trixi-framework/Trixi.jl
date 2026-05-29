
using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")

@testset "Wave Equations 1D" begin
#! format: noindent

@trixi_testset "elixir_wave_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_wave_convergence.jl"),
                        l2=[0.0027356242067981735, 0.002735624132579398],
                        linf=[0.010396415748525678, 0.01039641615909015])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end
