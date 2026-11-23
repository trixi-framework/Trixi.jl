using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation on [0, 1], periodic, with a
# smooth initial condition of two Fourier modes

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

function initial_condition_two_modes(x, t, equations::LinearScalarAdvectionEquation1D)
    x_trans = x[1] - equations.advection_velocity[1] * t
    return SVector(sinpi(2 * x_trans) + 0.5f0 * cospi(4 * x_trans))
end
initial_condition = initial_condition_two_modes

# Macro-cell finite volume solver of fifth order: WENO5-Z reconstruction of the subcell
# averages. The initial condition is set to the subcell averages (Gauss-Legendre
# quadrature with 5 nodes per subcell); point values at the subcell centers would limit
# the accuracy to second order.
#
# Note: the errors of the `AnalysisCallback` compare the subcell averages with the
# point values at the subcell centers and are therefore only second-order accurate.
n_cells = 8
reconstruction = ReconstructionWENO5Z()
solver = MacroCellFV(n_cells = n_cells, surface_flux = flux_lax_friedrichs,
                     reconstruction = reconstruction,
                     initial_condition_nodes = 5)

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

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
