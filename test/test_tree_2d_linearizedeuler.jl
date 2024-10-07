
using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_2d_dgsem")

@testset "Linearized Euler Equations 2D" begin
#! format: noindent

@trixi_testset "elixir_linearizedeuler_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_linearizedeuler_convergence.jl"),
                        l2=[
                            0.00020601485381444888,
                            0.00013380483421751216,
                            0.0001338048342174503,
                            0.00020601485381444888
                        ],
                        linf=[
                            0.0011006084408365924,
                            0.0005788678074691855,
                            0.0005788678074701847,
                            0.0011006084408365924
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
                        l2=[
                            0.048185623945503485,
                            0.01941899333212175,
                            0.019510224816991825,
                            0.048185623945503485
                        ],
                        linf=[
                            1.0392165942153189,
                            0.18188777290819994,
                            0.1877028372108587,
                            1.0392165942153189
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
