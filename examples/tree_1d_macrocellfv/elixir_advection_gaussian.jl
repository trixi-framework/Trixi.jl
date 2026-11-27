using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation on [0, 1], periodic,
# transporting a smooth Gaussian pulse

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

function initial_condition_gaussian_periodic(x, t,
                                             equations::LinearScalarAdvectionEquation1D)
    a = equations.advection_velocity[1]
    # Position relative to the transported pulse center (initially at x = 1/2),
    # wrapped into [-1/2, 1/2)
    xi = mod(x[1] - a * t, 1) - 0.5f0
    return SVector(exp(-(xi / 0.1f0)^2))
end
initial_condition = initial_condition_gaussian_periodic

# First-order macro-cell finite volume solver: identity moment inversion,
# piecewise constant reconstruction, and the upwind (Lax-Friedrichs) flux
n_cells = 8
solver = MacroCellFV(n_cells = n_cells, surface_flux = flux_lax_friedrichs,
                     reconstruction = ReconstructionConstant())

coordinates_min = 0.0
coordinates_max = 1.0
initial_refinement_level = 3
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK33();
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
