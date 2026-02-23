
using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")

@testset "Wave Equations 2D" begin
#! format: noindent

@trixi_testset "elixir_wave_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_wave_convergence.jl"),
                        l2=[
                            9.47185258e-05,
                            6.92245498e-05,
                            3.73189845e-05
                        ],
                        linf=[
                            3.73557666e-04,
                            2.92925075e-04,
                            7.90116854e-05
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end
