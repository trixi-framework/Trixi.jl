using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the pure diffusion equation with Dirichlet BCs (LDG)

diffusivity = 0.25
amplitude = 1.0
wave_number = pi

equations = LinearDiffusionEquation1D(diffusivity)

solver = DGSEM(polydeg = 3)
solver_parabolic = ParabolicFormulationLocalDG()

mesh = TreeMesh((0.0,), (1.0,),
                initial_refinement_level = 3,
                periodicity = false,
                n_cells_max = 30_000)

# Initial condition consistent with Dirichlet BCs and exact solution
initial_condition = (x, t, equations) -> SVector(amplitude *
                                                 exp(-diffusivity * wave_number^2 * t) *
                                                 sin(wave_number * x[1]))

boundary_condition_dirichlet = BoundaryConditionDirichlet((x, t, equations) -> SVector(0.0))

boundary_conditions = (; x_neg = boundary_condition_dirichlet,
                       x_pos = boundary_condition_dirichlet)

semi = SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                   solver_parabolic = solver_parabolic,
                                   boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 200, analysis_integrals = ())

alive_callback = AliveCallback(analysis_interval = 200)

stepsize_callback = StepsizeCallback(cfl_parabolic = 0.05)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = stepsize_callback(ode), adaptive = false,
            ode_default_options()..., callback = callbacks)
