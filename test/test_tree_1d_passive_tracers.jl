module TestExamples1DEuler

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Passive Tracers Tree 1D" begin
#! format: noindent

@trixi_testset "elixir_euler_density_wave_tracers.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave_tracers.jl"),
                        l2=[
                            0.07817688029733633,
                            0.007817688029733637,
                            0.0003908844014910887,
                            0.2559175560899598,
                            0.2855426873174796
                        ],
                        linf=[
                            0.23661504279664292,
                            0.023661504279667844,
                            0.0011830752140795653,
                            0.917972951035074,
                            0.9937748476119836
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
