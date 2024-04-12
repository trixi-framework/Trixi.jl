
using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_3d_dgsem")

@testset "Linearized Euler Equations 3D" begin
#! format: noindent

@trixi_testset "elixir_linearizedeuler_gauss_wall.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_linearizedeuler_gauss_wall.jl"),
                        l2=[
                            0.054201508303202306, 0.10347277056092179,
                            8.10542634693489e-17, 8.273108096127844e-17,
                            0.054201508303202306,
                        ],
                        linf=[
                            0.1774484851727698, 0.33221022902054875,
                            1.9884884692276712e-15, 1.641012595221666e-15,
                            0.1774484851727698,
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
