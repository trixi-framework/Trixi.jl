module TestExamplesParabolic1D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "SemidiscretizationHyperbolicParabolic (1D)" begin
#! format: noindent

@trixi_testset "TreeMesh1D: elixir_advection_diffusion.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        initial_refinement_level=4, tspan=(0.0, 0.4), polydeg=3,
                        l2=[8.40483031802723e-6],
                        linf=[2.8990878868540015e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh1D: elixir_advection_diffusion_restart.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem",
                                 "elixir_advection_diffusion_restart.jl"),
                        l2=[1.0679933947301556e-5],
                        linf=[3.910500545667439e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh1D: elixir_advection_diffusion.jl (AMR)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem",
                                 "elixir_advection_diffusion.jl"),
                        tspan=(0.0, 0.0), initial_refinement_level=5)
    tspan = (0.0, 1.0)
    ode = semidiscretize(semi, tspan)
    amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                          base_level = 4,
                                          med_level = 5, med_threshold = 0.1,
                                          max_level = 6, max_threshold = 0.6)
    amr_callback = AMRCallback(semi, amr_controller,
                               interval = 5,
                               adapt_initial_condition = true)

    # Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
    callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                            amr_callback)
    sol = solve(ode, KenCarp4(autodiff = AutoFiniteDiff());
                abstol = time_abs_tol, reltol = time_int_tol,
                ode_default_options()..., callback = callbacks)
    l2_error, linf_error = analysis_callback(sol)
    @test l2_error ≈ [6.487940740394583e-6]
    @test linf_error ≈ [3.262867898701227e-5]
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_periodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        l2=[
                            0.0001133835907077494,
                            6.226282245610444e-5,
                            0.0002820171699999139
                        ],
                        linf=[
                            0.0006255102377159538,
                            0.00036195501456059986,
                            0.0016147729485886941
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

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_periodic.jl: GradientVariablesEntropy" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[
                            0.00011310615871043463,
                            6.216495207074201e-5,
                            0.00028195843110817814
                        ],
                        linf=[
                            0.0006240837363233886,
                            0.0003616694320713876,
                            0.0016147339542413874
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

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_walls.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls.jl"),
                        l2=[
                            0.0004702331100298379,
                            0.0003218173539588441,
                            0.001496626616191212
                        ],
                        linf=[
                            0.0029963751636357117,
                            0.0028639041695096433,
                            0.012691132694550689
                        ],
                        atol=1e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_walls.jl: GradientVariablesEntropy" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[
                            0.00046085004909354776,
                            0.0003243109084492897,
                            0.0015159733164383632
                        ],
                        linf=[
                            0.0027548031865172184,
                            0.0028567713569609024,
                            0.012941793735691931
                        ],
                        atol=1e-9)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_walls_amr.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls_amr.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number()),
                        l2=[
                            2.5278845598681636e-5,
                            2.5540145802666872e-5,
                            0.0001211867535580826
                        ],
                        linf=[
                            0.0001466387202588848,
                            0.00019422419092429135,
                            0.0009556449835592673
                        ],
                        atol=1e-9)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_walls_amr.jl: GradientVariablesEntropy" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls_amr.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[
                            2.4593521887223632e-5,
                            2.3928212900127102e-5,
                            0.00011252332663824173
                        ],
                        linf=[
                            0.00011850494672183132,
                            0.00018987676556476442,
                            0.0009597423024825247
                        ],
                        atol=1e-9)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh1D: elixir_navierstokes_viscous_shock.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        l2=[
                            0.00025762354103445303,
                            0.0001433692781569829,
                            0.00017369861968287976
                        ],
                        linf=[
                            0.0016731940030498826,
                            0.0010638575921477766,
                            0.0011495207677434394
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

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)

end # module
