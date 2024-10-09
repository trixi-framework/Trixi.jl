# The same setup as tree_2d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh

using OrdinaryDiffEq
using Trixi
using Trixi2Vtk


function initial_constant(x, t, equations)
    return SVector(2.5,)
end

function initial_sinx(x, t, equations)
    return SVector(sin(x[1]*pi),)
end

function initial_siny(x, t, equations)
    return SVector(sin(x[2]*pi),)
end

function initial_linx(x, t, equations)
    return SVector(x[1],)
end

function initial_liny(x, t, equations)
    return SVector(x[2],)
end

function gaussian(x, t, equations)
    if x[1] < 0
        return SVector(exp((-(x[1]+0.5)^2 - x[2]^2)*15),)
    else
        return SVector(0.0,)
    end
end

initial_condition = gaussian

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (1.0, 0.0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)
#                volume_integral=VolumeIntegralUpwind(flux_lax_friedrichs))

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

trees_per_dimension = (8, 8)

# Create P4estMesh with 8 x 8 trees and 16 x 16 elements
parent_mesh = P4estMesh(trees_per_dimension, polydeg=3,
                        coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                        initial_refinement_level=1, periodicity=(false, true))
mesh1 = P4estMeshView(parent_mesh; indices_min = (1, 1), indices_max = (6, 8),
                      coordinates_min = coordinates_min, coordinates_max = (0.5, 1.0),
                      periodicity = (false, true))
mesh2 = P4estMeshView(parent_mesh; indices_min = (7, 1), indices_max = (8, 8),
                      coordinates_min = (0.5, -1.0), coordinates_max = coordinates_max,
                      periodicity = (false, true))

coupling_function1 = (x, u, equations_other, equations_own) -> u
coupling_function2 = (x, u, equations_other, equations_own) -> u

boundary_conditions1 = Dict(:x_neg => BoundaryConditionCoupled(2, (:end, :i_forward), Float64, coupling_function1),
                            :x_pos => BoundaryConditionCoupled(2, (:begin, :i_forward), Float64, coupling_function1))
boundary_conditions2 = Dict(:x_neg => BoundaryConditionCoupled(1, (:end, :i_forward), Float64, coupling_function2),
                            :x_pos => BoundaryConditionCoupled(1, (:begin, :i_forward), Float64, coupling_function2))

# A semidiscretization collects data structures and functions for the spatial discretization
semi1 = SemidiscretizationHyperbolic(mesh1, equations, initial_condition, solver, boundary_conditions=boundary_conditions1)
semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition, solver, boundary_conditions=boundary_conditions2)

# Create a semidiscretization that bundles semi1 and semi2
semi = SemidiscretizationCoupled(semi1, semi2)

###############################################################################
# ODE solvers, callbacks etc.

# ode = semidiscretize(semi, (0.0, 20.3));
ode = semidiscretize(semi, (0.0, 5.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback1 = AnalysisCallback(semi1, interval=100)
analysis_callback2 = AnalysisCallback(semi2, interval=100)
analysis_callback = AnalysisCallbackCoupled(semi, analysis_callback1, analysis_callback2)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=1,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

# Convert the snapshots into vtk format.
# trixi2vtk("out/solution_*.h5")
