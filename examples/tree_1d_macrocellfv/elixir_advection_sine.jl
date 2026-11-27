using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation
#   u_t + a u_x = 0  on [0, 1], periodic,
# with the exact solution u(x, t) = sin(2 pi (x - a t)).

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

function initial_condition_sine(x, t, equations::LinearScalarAdvectionEquation1D)
    a = equations.advection_velocity[1]
    return SVector(sinpi(2 * (x[1] - a * t)))
end
initial_condition = initial_condition_sine

# Macro-cell finite volume solver with `n_cells` subcells per macro-cell, MUSCL
# reconstruction with unlimited central slopes (second order; e.g.,
# `slope_limiter = minmod` for a TVD scheme), and the (local) Lax-Friedrichs/Rusanov
# flux
n_cells = 8
slope_limiter = central_slope
solver = MacroCellFV(n_cells = n_cells, surface_flux = flux_lax_friedrichs,
                     reconstruction = ReconstructionMUSCL(slope_limiter))

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

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:l2_error_primitive,
                                                              :linf_error_primitive,
                                                              :conservation_error))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK33();
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
