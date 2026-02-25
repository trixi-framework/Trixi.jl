
using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")

@testset "Wave Equations 2D" begin
#! format: noindent

@trixi_testset "elixir_wave_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_wave_convergence.jl"),
                        l2=[
                            2.4292905839503593e-5,
                            3.1433013845821566e-5,
                            1.588526532948571e-5
                        ],
                        linf=[
                            0.00016841204839490587,
                            0.00016464766108787723,
                            4.139321166533423e-5
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end
