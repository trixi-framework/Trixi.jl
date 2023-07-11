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
mapping = Trixi.transfinite_mapping(faces)

# Create P4estMesh with 3 x 2 trees and 6 x 4 elements,
# approximate the geometry with a smaller polydeg for testing.
trees_per_dimension = (3, 2)
mesh = T8codeMesh(trees_per_dimension, polydeg = 3,
                  mapping = mapping,
                  initial_refinement_level = 1)

function adapt_callback(forest,
                        forest_from,
                        which_tree,
                        lelement_id,
                        ts,
                        is_family,
                        num_elements,
                        elements_ptr)::Cint
    vertex = Vector{Cdouble}(undef, 3)

    elements = unsafe_wrap(Array, elements_ptr, num_elements)

    Trixi.t8_element_vertex_reference_coords(ts, elements[1], 0, pointer(vertex))

    level = Trixi.t8_element_level(ts, elements[1])

    # TODO: Make this condition more general.
    if vertex[1] < 1e-8 && vertex[2] < 1e-8 && level < 4
        # return true (refine)
        return 1
    else
        # return false (don't refine)
        return 0
    end
end

Trixi.@T8_ASSERT(Trixi.t8_forest_is_committed(mesh.forest)!=0);

# Init new forest.
new_forest_ref = Ref{Trixi.t8_forest_t}()
Trixi.t8_forest_init(new_forest_ref);
new_forest = new_forest_ref[]

# Check out `examples/t8_step4_partition_balance_ghost.jl` in
# https://github.com/DLR-AMR/T8code.jl for detailed explanations.
let set_from = C_NULL, recursive = 1, set_for_coarsening = 0, no_repartition = 0
    Trixi.t8_forest_set_user_data(new_forest, C_NULL)
    Trixi.t8_forest_set_adapt(new_forest, mesh.forest,
                              Trixi.@t8_adapt_callback(adapt_callback), recursive)
    Trixi.t8_forest_set_balance(new_forest, set_from, no_repartition)
    Trixi.t8_forest_set_partition(new_forest, set_from, set_for_coarsening)
    Trixi.t8_forest_set_ghost(new_forest, 1, Trixi.T8_GHOST_FACES)
    Trixi.t8_forest_commit(new_forest)
end

mesh.forest = new_forest

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
