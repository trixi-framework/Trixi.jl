using Trixi
using OrdinaryDiffEq
using Plots

cs = 340.0
N = 0.01
equations = LinearizedGravityWaveEquations2D(cs, N)

solver = DGSEM(polydeg = 1, surface_flux = flux_godunov)

boundary_conditions = (
    x_neg = boundary_condition_periodic,
    x_pos = boundary_condition_periodic,
    y_neg = boundary_condition_slip_wall,
    y_pos = boundary_condition_slip_wall,
)

coordinates_min = (-150_000.0, 0.0)
coordinates_max = (150_000.0, 10_000.0)

cells_per_dimension = (128, 128)

mesh = StructuredMesh(
    cells_per_dimension,
    coordinates_min,
    coordinates_max,
    periodicity = (true, false),
)

initial_condition = initial_condition_convergence_test
semi = SemidiscretizationHyperbolic(
    mesh,
    equations,
    initial_condition,
    solver,
    boundary_conditions = boundary_conditions, source_terms = source_terms_convergence_test
)

tspan = (0.0, 200.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback =
    AnalysisCallback(semi, interval = analysis_interval, save_analysis = true)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(
    interval = analysis_interval,
    save_initial_solution = true,
    save_final_solution = true,
    solution_variables = cons2prim,
)

stepsize_callback = StepsizeCallback(cfl = 0.1)

callbacks = CallbackSet(
    summary_callback,
    analysis_callback,
    alive_callback,
    save_solution,
    stepsize_callback,
)

###############################################################################
# run the simulation
sol = solve(
    ode,
    CarpenterKennedy2N54(williamson_condition = false),
    maxiters = 1.0e7,
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    save_everystep = false,
    callback = callbacks,
);
summary_callback() # print the timer summary
pd = PlotData2D(sol)
plot(pd["b"], aspect_ratio = 3)