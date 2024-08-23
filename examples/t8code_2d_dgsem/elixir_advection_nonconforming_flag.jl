using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 4 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-1.0, s - 1.0)
f2(s) = SVector(1.0, s + 1.0)
f3(s) = SVector(s, -1.0 + sin(0.5 * pi * s))
f4(s) = SVector(s, 1.0 + sin(0.5 * pi * s))

faces = (f1, f2, f3, f4)

# Create T8codeMesh with 3 x 2 trees and 6 x 4 elements,
# approximate the geometry with a smaller polydeg for testing.
trees_per_dimension = (3, 2)
mesh = T8codeMesh(trees_per_dimension, polydeg = 3,
                  faces = faces,
                  initial_refinement_level = 1,
                  periodicity = (true, true))

# Note: This is actually a `p4est_quadrant_t` which is much bigger than the
# following struct. But we only need the first three fields for our purpose.
struct t8_dquad_t
    x::Int32
    y::Int32
    level::Int8
    # [...] # See `p4est.h` in `p4est` for more info.
end

# Refine quadrants of each tree at lower left edge to level 4.
function adapt_callback(forest, ltreeid, eclass_scheme, lelemntid, elements, is_family,
                        user_data)
    el = unsafe_load(Ptr{t8_dquad_t}(elements[1]))

    if el.x == 0 && el.y == 0 && el.level < 4
        # return true (refine)
        return 1
    else
        # return false (don't refine)
        return 0
    end
end

Trixi.adapt!(mesh, adapt_callback)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 0.2
ode = semidiscretize(semi, (0.0, 0.2));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()

# Finalize `T8codeMesh` to make sure MPI related objects in t8code are
# released before `MPI` finalizes.
!isinteractive() && finalize(mesh)
