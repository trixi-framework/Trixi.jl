module TestExamples1DEuler

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "p4est_2d_dgsem")

@testset "Passive Tracers Tree 2D" begin
#! format: noindent

@trixi_testset "elixir_euler_density_wave_tracers.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave_tracers.jl"),
                        l2=[
                            0.0012704690524147188, 0.00012704690527390463,
                            0.00025409381047976197, 3.17617263147723e-5,
                            0.00134504046174889, 0.0017674381603697698
                        ],
                        linf=[
                            0.0071511674295154926, 0.0007151167435655859,
                            0.0014302334865533006, 0.00017877918656949987,
                            0.005363513244351692, 0.011878722952669549
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end
end # testset
end # module
