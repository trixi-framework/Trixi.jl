using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0

"""
    initial_condition_composite(x, t, equations::LinearScalarAdvectionEquation1D)

Wave form that is a combination of a Gaussian pulse, a square wave, a triangle wave,
and half an ellipse with periodic boundary conditions.
Slight simplification from
- Jiang, Shu (1996)
  Efficient Implementation of Weighted ENO Schemes
  [DOI: 10.1006/jcph.1996.0130](https://doi.org/10.1006/jcph.1996.0130)
"""
function initial_condition_composite(x, t, equations::LinearScalarAdvectionEquation1D)
    xmin, xmax = -1.0, 1.0 # Only works if the domain is [-1.0,1.0]
    x_trans = x[1] - t
    L = xmax - xmin
    if x_trans > xmax
        x_ = x_trans - L * floor((x_trans + xmin) / L)
    elseif x_trans < xmin
        x_ = x_trans + L * floor((xmax - x_trans) / L)
    else
        x_ = x_trans
    end

    if x_ > -0.8 && x_ < -0.6
        value = exp(-log(2.0) * (x_ + 0.7)^2 / 0.0009)
    elseif x_ > -0.4 && x_ < -0.2
        value = 1.0
    elseif x_ > 0.0 && x_ < 0.2
        value = 1.0 - abs(10.0 * (x_ - 0.1))
    elseif x_ > 0.4 && x_ < 0.6
        value = sqrt(1.0 - 100.0 * (x_ - 0.5)^2)
    else
        value = 0.0
    end

    return SVector(value)
end

initial_condition = initial_condition_composite

equations = LinearScalarAdvectionEquation1D(advection_velocity)

surface_flux = flux_lax_friedrichs
volume_flux = flux_central
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = Trixi.first)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# Create curved mesh
cells_per_dimension = (120,)
coordinates_min = (-1.0,) # minimum coordinate
coordinates_max = (1.0,)  # maximum coordinate
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max,
                      periodicity = true)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with a given time span
tspan = (0.0, 4.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100, solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.5)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
