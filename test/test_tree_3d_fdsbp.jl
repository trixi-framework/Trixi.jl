module TestTree3DFDSBP

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_3d_fdsbp")

@testset "Linear scalar advection" begin
#! format: noindent

@trixi_testset "elixir_advection_extended.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                        l2=[0.005355755365412444],
                        linf=[0.01856044696350767])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_extended.jl with periodic operators" begin
    global D = SummationByPartsOperators.periodic_derivative_operator(derivative_order = 1,
                                                                      accuracy_order = 4,
                                                                      xmin = 0.0,
                                                                      xmax = 1.0,
                                                                      N = 10)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                        l2=[5.228248923012878e-9],
                        linf=[9.24430243465224e-9],
                        D_SBP=D,
                        initial_refinement_level=0,
                        tspan=(0.0, 5.0))

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

@testset "Compressible Euler" begin
    @trixi_testset "elixir_euler_convergence.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
                            l2=[
                                2.247522803543667e-5,
                                2.2499169224681058e-5,
                                2.24991692246826e-5,
                                2.2499169224684707e-5,
                                5.814121361417382e-5
                            ],
                            linf=[
                                9.579357410749445e-5,
                                9.544871933409027e-5,
                                9.54487193367548e-5,
                                9.544871933453436e-5,
                                0.0004192294529472562
                            ],
                            tspan=(0.0, 0.2))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_convergence.jl with VolumeIntegralStrongForm" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
                            l2=[
                                4.084919840272202e-5,
                                4.1320630860402814e-5,
                                4.132063086040211e-5,
                                4.132063086039092e-5,
                                8.502518355874354e-5
                            ],
                            linf=[
                                0.0001963934848161486,
                                0.00020239883896255861,
                                0.0002023988389729947,
                                0.00020239883896766564,
                                0.00052605624510349
                            ],
                            tspan=(0.0, 0.2),
                            solver=DG(D_upw.central, nothing, SurfaceIntegralStrongForm(),
                                      VolumeIntegralStrongForm()))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_taylor_green_vortex.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_taylor_green_vortex.jl"),
                            l2=[
                                3.529693407280806e-6,
                                0.0004691301922633193,
                                0.00046913019226332234,
                                0.0006630180220973541,
                                0.0015732759680929076
                            ],
                            linf=[
                                3.4253965106145756e-5,
                                0.0010033197685090707,
                                0.0010033197685091054,
                                0.0018655642702542635,
                                0.008479800046757191
                            ],
                            tspan=(0.0, 0.0075), abstol=1.0e-9, reltol=1.0e-9)

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

end # module
