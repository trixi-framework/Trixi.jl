
using OrdinaryDiffEq
using Trixi

###############################################################################
# Couple polytropic Euler systems.
#
# In a rectangular domain we solve two different sets of equations for the
# left and the right half. Those are the isotropic equations (left)
# and polytropic equations (right).
# The coupling hapens on two interfaces. One is located at the center of the
# domain such that the right half is coupled through its left boundary
# and the left half is coupled through its right boundary. The other coupling
# makes sure that the domain is periodic. Here the right boundary of the right
# domain is coupled to the left boundary of the left domain.
# The vertical boundaries are simply periodic.
# As test case we send a linear wave through the domain and observe a change
# in the dispersion relation when the wave enters the isotropic domain.

###############################################################################
# define the initial conditions

function initial_condition_wave_isotropic(x, t, equations::PolytropicEulerEquations2D)
    gamma = 1.001
    kappa = 1.0

    rho = 1.0
    v1 = 0.0
    v2 = 0.0

    return prim2cons(SVector(rho, v1, v2), equations)
end

function initial_condition_wave_polytropic(x, t, equations::PolytropicEulerEquations2D)
    gamma = 2.0
    kappa = 1.0

    rho = 1.0
    v1 = 0.0
    if x[1] > 0.0
        rho = ((1.0 + 0.01 * sin(x[1] * 2 * pi)) / kappa)^(1 / gamma)
        v1 = ((0.01 * sin((x[1] - 1 / 2) * 2 * pi)) / kappa)
    end
    v2 = 0.0

    return prim2cons(SVector(rho, v1, v2), equations)
end

###############################################################################
# semidiscretization of the linear advection equation

function Trixi.wrap_array(u_ode::AbstractVector, mesh::Trixi.AbstractMesh, equations,
                          dg::DGSEM, cache)
    Trixi.wrap_array_native(u_ode, mesh, equations, dg, cache)
end

# Define the equations involved.
gamma1 = 1.001
kappa1 = 1.0
equations1 = PolytropicEulerEquations2D(gamma_A, kappa_A)
gamma2 = 2.0
kappa2 = 1.0
equations2 = PolytropicEulerEquations2D(gamma_B, kappa_B)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
volume_flux = flux_ranocha
solver = DGSEM(polydeg = 2, surface_flux = flux_hll,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min1 = (-2.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max1 = (0.0, 1.0) # maximum coordinates (max(x), max(y))
coordinates_min2 = (0.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max2 = (2.0, 1.0) # maximum coordinates (max(x), max(y))

cells_per_dimension = (32, 32)

mesh1 = StructuredMesh(cells_per_dimension,
                       coordinates_min1,
                       coordinates_max1)
mesh2 = StructuredMesh(cells_per_dimension,
                       coordinates_min2,
                       coordinates_max2)

# A semidiscretization collects data structures and functions for the spatial discretization.
semi1 = SemidiscretizationHyperbolic(mesh1, equations1,
                                     initial_condition_wave_isotropic, solver,
                                     boundary_conditions = (x_neg = BoundaryConditionCoupled(2,
                                                                                             (:end,
                                                                                              :i_forward),
                                                                                             Float64),
                                                            x_pos = BoundaryConditionCoupled(2,
                                                                                             (:begin,
                                                                                              :i_forward),
                                                                                             Float64),
                                                            y_neg = boundary_condition_periodic,
                                                            y_pos = boundary_condition_periodic))

semi2 = SemidiscretizationHyperbolic(mesh2, equations2,
                                     initial_condition_wave_polytropic, solver,
                                     boundary_conditions = (x_neg = BoundaryConditionCoupled(1,
                                                                                             (:end,
                                                                                              :i_forward),
                                                                                             Float64),
                                                            x_pos = BoundaryConditionCoupled(1,
                                                                                             (:begin,
                                                                                              :i_forward),
                                                                                             Float64),
                                                            y_neg = boundary_condition_periodic,
                                                            y_pos = boundary_condition_periodic))

# Create a semidiscretization that bundles semi1 and semi2
semi = SemidiscretizationCoupled(semi1, semi2)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 30.0
ode = semidiscretize(semi, (0.0, 3.0))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

resid_tol = 5.0e-12
steady_state_callback = SteadyStateCallback(abstol = resid_tol, reltol = 0.0)

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback1 = AnalysisCallback(semi1, interval = 100)
analysis_callback2 = AnalysisCallback(semi2, interval = 100)
analysis_callback = AnalysisCallbackCoupled(semi, analysis_callback1, analysis_callback2)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 5,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 1.0)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()

