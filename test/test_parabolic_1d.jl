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
                        l2=[8.389498188525518e-06],
                        linf=[2.847421658558336e-05])
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
    sol = solve(ode, KenCarp4(autodiff = false), abstol = time_abs_tol,
                reltol = time_int_tol,
                save_everystep = false, callback = callbacks)
    l2_error, linf_error = analysis_callback(sol)
    @test l2_error ≈ [6.4878111416468355e-6]
    @test linf_error ≈ [3.258075790424364e-5]
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
                            0.00047023310868269237,
                            0.00032181736027057234,
                            0.0014966266486095025
                        ],
                        linf=[
                            0.002996375101363302,
                            0.0028639041695096433,
                            0.012691132694550689
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

@trixi_testset "TreeMesh1D: elixir_navierstokes_convergence_walls.jl: GradientVariablesEntropy" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem",
                                 "elixir_navierstokes_convergence_walls.jl"),
                        equations_parabolic=CompressibleNavierStokesDiffusion1D(equations,
                                                                                mu = mu(),
                                                                                Prandtl = prandtl_number(),
                                                                                gradient_variables = GradientVariablesEntropy()),
                        l2=[
                            0.0004608500483647771,
                            0.00032431091222851285,
                            0.0015159733360626845
                        ],
                        linf=[
                            0.002754803146635787,
                            0.0028567713744625124,
                            0.012941793784197131
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
                            0.0009597461727750556
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
