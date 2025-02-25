
using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_3d_dgsem")

@testset "Linearized Euler Equations 3D" begin
#! format: noindent

@trixi_testset "elixir_linearizedeuler_gauss_wall.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_linearizedeuler_gauss_wall.jl"),
                        l2=[
                            0.020380328336745232, 0.027122442311921492,
                            0.02712244231192152, 8.273108096127844e-17,
                            0.020380328336745232
                        ],
                        linf=[
                            0.2916021983572774, 0.32763703462270843,
                            0.32763703462270855, 1.641012595221666e-15,
                            0.2916021983572774
                        ],
                        tspan=(0.0, 1.0))

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
