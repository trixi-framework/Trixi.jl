using OrdinaryDiffEqLowStorageRK
using Trixi
using Measurements # For propagating uncertainty/measurement errors in parameters

###############################################################################
# semidiscretization of the linear advection equation

# Note the `±` operator for defining uncertain parameters
advection_velocity = 1.0 ± 0.1
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0 # minimum coordinate
coordinates_max = 1.0 # maximum coordinate
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                periodicity = true)

# The solution inherits the uncertainty of the advection velocity
uEltype = Measurement{Float64}
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver;
                                    boundary_conditions = boundary_condition_periodic,
                                    uEltype = uEltype)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 50)

callbacks = CallbackSet(summary_callback, analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49();
            ode_default_options()..., callback = callbacks);
