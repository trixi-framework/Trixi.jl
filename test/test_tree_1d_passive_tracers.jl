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
                            4.01591251149393e-5, 4.015912637380159e-6,
                            2.0079560103160982e-7, 4.026204610120812e-5,
                            4.076015241597719e-5
                        ],
                        linf=[
                            0.00014752217918240218, 1.4752217787357413e-5,
                            7.376111597068302e-7, 0.0001523957753177818,
                            0.00016524903474501862
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
