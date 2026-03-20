using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the pure diffusion equation with pure Dirichlet BCs (LDG)

diffusivity = 0.5
amplitude = 0.4
wave_number = pi

equations = LinearDiffusionEquation1D(diffusivity)

solver = DGSEM(polydeg = 3, surface_flux = flux_central)
solver_parabolic = ParabolicFormulationLocalDG()

mesh = TreeMesh((0.0,), (1.0,),
                initial_refinement_level = 3,
                periodicity = false,
                n_cells_max = 30_000)

initial_condition = (x, t, equations) -> SVector(amplitude * exp(-diffusivity * wave_number^2 * t) * sin(wave_number * x[1]))

boundary_condition_dirichlet = BoundaryConditionDirichlet((x, t, equations) -> SVector(0.0))

boundary_conditions = (; x_neg = boundary_condition_dirichlet,
                       x_pos = boundary_condition_dirichlet)

semi = SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                   solver_parabolic = solver_parabolic,
                                   boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_integrals = ())

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl_parabolic = 0.1)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = stepsize_callback(ode), adaptive = false,
            ode_default_options()..., callback = callbacks)
