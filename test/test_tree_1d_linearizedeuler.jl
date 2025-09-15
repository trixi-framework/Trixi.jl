
using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Linearized Euler Equations 1D" begin
#! format: noindent

@trixi_testset "elixir_linearizedeuler_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_linearizedeuler_convergence.jl"),
                        l2=[
                            0.00010894927270421941,
                            0.00014295255695912358,
                            0.00010894927270421941
                        ],
                        linf=[
                            0.0005154647164193893,
                            0.00048457837684242266,
                            0.0005154647164193893
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

@trixi_testset "elixir_linearizedeuler_gauss_wall.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_linearizedeuler_gauss_wall.jl"),
                        l2=[0.650082087850354, 0.2913911415488769, 0.650082087850354],
                        linf=[
                            1.9999505145390108,
                            0.9999720404625275,
                            1.9999505145390108
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
end
