using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (3.0, 3.0)
diffusivity() = 5.0e-2
equations = LinearScalarAdvectionEquation2D(advection_velocity)
equations_parabolic = LaplaceDiffusion2D(diffusivity(), equations)

function initial_condition_gauss_static(x, t, equations)
    # center of the Gaussian (choose safely inside your domain)
    x0 = -0.5
    y0 = 1.5

    # width (smaller = more localized)
    sigma = 0.175

    # Gaussian
    r2 = (x[1] - x0)^2 + (x[2] - y0)^2
    u = 2 * exp(-r2 / (2 * sigma^2))

    return SVector(u)
end

initial_condition = initial_condition_gauss_static

boundary_condition = BoundaryConditionDirichlet((x, t, equations) -> SVector(0.0))
boundary_conditions = (Bottom = boundary_condition,
                       Right = boundary_condition,
                       Top = boundary_condition,
                       Left = boundary_condition)
boundary_conditions_parabolic = boundary_conditions

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# Unstructured mesh with inverted indexing of the nodes
# Specifically tests that the node indexing in the mpi parabolic solver is consistent 
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/3e7dac35eeadc24739ea29619f78b8d2/raw/ecc29e82be69c656217dc878821917850c51e41d/mesh_wobbly_channel.inp",
                           joinpath(@__DIR__, "mesh_wobbly_channel.inp"))

mesh = P4estMesh{2}(mesh_file, polydeg = 3,
                    initial_refinement_level = 1)

semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = ParabolicFormulationBassiRebay1(),
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                      base_level = 1,
                                      med_level = 2, med_threshold = 0.05,
                                      max_level = 3, max_threshold = 0.3)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart, save_solution,
                        amr_callback);

###############################################################################
# run the simulation
ode_alg = RDPK3SpFSAL49()
sol = solve(ode, ode_alg;
            dt = 1e-5, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
