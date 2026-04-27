
using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")

@testset "Wave Equations 2D" begin
#! format: noindent

@trixi_testset "elixir_wave_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_wave_convergence.jl"),
                        l2=[
                            2.4495172317756833e-5,
                            3.174909812221007e-5,
                            2.9979271536156852e-6
                        ],
                        linf=[
                            0.00016911867790070367,
                            0.0001640678341614522,
                            1.6269923975083337e-5
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end
