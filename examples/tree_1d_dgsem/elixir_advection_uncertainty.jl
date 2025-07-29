using Trixi
using OrdinaryDiffEqLowStorageRK
using Measurements # For propagating uncertainty/measurement errors in parameters

# Note the `±` operator for defining uncertain parameters
equations = LinearScalarAdvectionEquation1D(1.0 ± 0.1)

x_min = (-1.0,)
x_max = (1.0,)
mesh = TreeMesh(x_min, x_max,
                n_cells_max = 10^5, initial_refinement_level = 5)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

RealT = Measurement{Float64} # Measurement datatype
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver, uEltype = RealT)

tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 50)

callbacks = CallbackSet(summary_callback, analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49();
            ode_default_options()..., callback = callbacks);

