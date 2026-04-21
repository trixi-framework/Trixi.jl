module TestTree1DFDSBP

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_fdsbp")

@testset "Linear scalar advection" begin
#! format: noindent

@trixi_testset "elixir_advection_upwind.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_upwind.jl"),
                        l2=[1.7735637157305526e-6],
                        linf=[1.0418854521951328e-5],
                        tspan=(0.0, 0.5))

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_upwind_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_upwind_periodic.jl"),
                        l2=[1.1672962783692568e-5],
                        linf=[1.650514414558435e-5])

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

@testset "Inviscid Burgers" begin
    @trixi_testset "elixir_burgers_basic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_basic.jl"),
                            l2=[8.316190308678742e-7],
                            linf=[7.1087263324720595e-6],
                            tspan=(0.0, 0.5))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    # same tolerances as above since the methods should be identical (up to
    # machine precision)
    @trixi_testset "elixir_burgers_basic.jl with SurfaceIntegralStrongForm and FluxUpwind" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_basic.jl"),
                            l2=[8.316190308678742e-7],
                            linf=[7.1087263324720595e-6],
                            tspan=(0.0, 0.5),
                            solver=DG(D_upw, nothing,
                                      SurfaceIntegralStrongForm(FluxUpwind(flux_splitting)),
                                      VolumeIntegralUpwind(flux_splitting)))
    end

    @trixi_testset "elixir_burgers_linear_stability.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_linear_stability.jl"),
                            l2=[0.9999995642691271],
                            linf=[1.824702804788453],
                            tspan=(0.0, 0.25))

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
                                4.1370344463620254e-6,
                                4.297052451817826e-6,
                                9.857382045003056e-6
                            ],
                            linf=[
                                1.675305070092392e-5,
                                1.3448113863834266e-5,
                                3.8185336878271414e-5
                            ],
                            tspan=(0.0, 0.5))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_convergence.jl with splitting_vanleer_haenel" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
                            l2=[
                                3.413790589105506e-6,
                                4.243957977156001e-6,
                                8.667369423676437e-6
                            ],
                            linf=[
                                1.4228079689537765e-5,
                                1.3249887941046978e-5,
                                3.201552933251861e-5
                            ],
                            tspan=(0.0, 0.5),
                            flux_splitting=splitting_vanleer_haenel)

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
                                8.6126767518378e-6,
                                7.670897071480729e-6,
                                1.4972772284191368e-5
                            ],
                            linf=[
                                6.707982777909294e-5,
                                3.487256699541419e-5,
                                0.00010170331350556339
                            ],
                            tspan=(0.0, 0.5),
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

    @trixi_testset "elixir_euler_density_wave.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
                            l2=[
                                1.5894925236031034e-5,
                                9.428412101106044e-6,
                                0.0008986477358789918
                            ],
                            linf=[
                                4.969438024382544e-5,
                                2.393091812063694e-5,
                                0.003271817388146303
                            ],
                            tspan=(0.0, 0.005), abstol=1.0e-9, reltol=1.0e-9)

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
