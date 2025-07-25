module TestExamplesStructuredMesh1D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "structured_1d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "StructuredMesh1D" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[6.0388296447998465e-6],
                        linf=[3.217887726258972e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic.jl"),
                        l2=[5.641921365468918e-5],
                        linf=[0.00021049780975179733])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_shockcapturing.jl"),
                        l2=[0.08015029105233593],
                        linf=[0.610709468736576],
                        atol=1.0e-5)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_float128.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_float128.jl"),
                        l2=Float128[6.49879312655540217059228636803492411e-09],
                        linf=Float128[5.35548407857266390181158920649552284e-08])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

# Testing the third-order paired explicit Runge-Kutta (PERK) method with its optimal CFL number
@trixi_testset "elixir_burgers_perk3.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_perk3.jl"),
                        l2=[3.8156922097242205e-6],
                        linf=[2.1962957979626552e-5],
                        atol=1.0e-6)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 8000
    end

    # Test `resize!`
    integrator = Trixi.init(ode, ode_algorithm, dt = 42.0, callback = callbacks)
    resize!(integrator, 42)
    @test length(integrator.u) == 42
    @test length(integrator.du) == 42
    @test length(integrator.u_tmp) == 42
    @test length(integrator.k1) == 42
    @test length(integrator.kS1) == 42
end

# Testing the third-order paired explicit Runge-Kutta (PERK) method without stepsize callback
@trixi_testset "elixir_burgers_perk3.jl(fixed time step)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_perk3.jl"),
                        dt=2.0e-3,
                        tspan=(0.0, 2.0),
                        save_solution=SaveSolutionCallback(dt = 0.1 + 1.0e-8), # Adding a small epsilon to avoid floating-point precision issues
                        callbacks=CallbackSet(summary_callback, save_solution,
                                              analysis_callback, alive_callback),
                        l2=[5.726144824784944e-7],
                        linf=[3.43073006914274e-6])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 8000
    end
end

@trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[3.67478226e-01, 3.49491179e-01, 8.08910759e-01],
                        linf=[1.58971947e+00, 1.59812384e+00, 1.94732969e+00],
                        tspan=(0.0, 0.3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_sedov_hll_davis.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[1.278661029299215, 0.0663853410742763, 0.9585741943783386],
                        linf=[
                            3.1661064228547255,
                            0.16256363944708607,
                            2.667676158812806
                        ],
                        tspan=(0.0, 12.5),
                        surface_flux=FluxHLL(min_max_speed_davis))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[
                            2.2527950196212703e-8,
                            1.8187357193835156e-8,
                            7.705669939973104e-8
                        ],
                        linf=[
                            1.6205433861493646e-7,
                            1.465427772462391e-7,
                            5.372255111879554e-7
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

@trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic.jl"),
                        l2=[
                            3.8099996914101204e-6,
                            1.6745575717106341e-6,
                            7.732189531480852e-6
                        ],
                        linf=[
                            1.2971473393186272e-5,
                            9.270328934274374e-6,
                            3.092514399671842e-5
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

@trixi_testset "elixir_euler_weak_blast_er.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_weak_blast_er.jl"),
                        analysis_interval=100,
                        l2=[0.1199630838410044,
                            0.1562196058317499,
                            0.44836353019483344],
                        linf=[
                            0.2255546997256792,
                            0.29412938937652194,
                            0.8558237244455227
                        ])
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15_000
    end

    # test both short and long printing formats
    @test_nowarn show(relaxation_solver)
    println()
    @test_nowarn println(relaxation_solver)
    println()
    @test_nowarn display(relaxation_solver)
    # Test `:compact` printing
    show(IOContext(IOBuffer(), :compact => true), MIME"text/plain"(), relaxation_solver)
end

@trixi_testset "elixir_linearizedeuler_characteristic_system.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_linearizedeuler_characteristic_system.jl"),
                        l2=[2.9318078842789714e-6, 0.0, 0.0],
                        linf=[4.291208715723194e-5, 0.0, 0.0])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_traffic_flow_lwr_greenlight.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_traffic_flow_lwr_greenlight.jl"),
                        l2=[0.2005523261652845],
                        linf=[0.5052827913468407])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_convergence_pure_fv.jl" begin
    @test_trixi_include(joinpath(pkgdir(Trixi, "examples", "tree_1d_dgsem"),
                                 "elixir_euler_convergence_pure_fv.jl"),
                        mesh=StructuredMesh(16, (0.0,), (2.0,)),
                        l2=[
                            0.019355699748523896,
                            0.022326984561234497,
                            0.02523665947241734
                        ],
                        linf=[
                            0.02895961127645519,
                            0.03293442484199227,
                            0.04246098278632804
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

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end # module
