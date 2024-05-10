
using OrdinaryDiffEq
using Trixi

# define new structs inside a module to allow re-evaluating the file
module TrixiExtension

using Trixi

struct IndicatorSolutionIndependent{Cache <: NamedTuple} <: Trixi.AbstractIndicator
    cache::Cache
end

function IndicatorSolutionIndependent(semi)
    basis = semi.solver.basis
    alpha = Vector{real(basis)}()
    cache = (; semi.mesh, alpha)
    return IndicatorSolutionIndependent{typeof(cache)}(cache)
end

function (indicator::IndicatorSolutionIndependent)(u::AbstractArray{<:Any, 4},
                                                   mesh, equations, dg, cache;
                                                   t, kwargs...)
    mesh = indicator.cache.mesh
    alpha = indicator.cache.alpha
    resize!(alpha, nelements(dg, cache))

    #Predict the theoretical center.
    advection_velocity = (0.2, -0.7)
    center = t .* advection_velocity

    inner_distance = 1
    outer_distance = 1.85

    #Iterate over all elements
    for element in eachindex(alpha)
        # Calculate periodic distance between cell and center.
        # This requires an uncurved mesh!
        coordinates = SVector(0.5 * (cache.elements.node_coordinates[1, 1, 1, element] +
                               cache.elements.node_coordinates[1, end, 1, element]),
                              0.5 * (cache.elements.node_coordinates[2, 1, 1, element] +
                               cache.elements.node_coordinates[2, 1, end, element]))

        #The geometric shape of the amr should be preserved when the base_level is increased.
        #This is done by looking at the original coordinates of each cell.
        cell_coordinates = original_coordinates(coordinates, 5 / 8)
        cell_distance = periodic_distance_2d(cell_coordinates, center, 10)
        if cell_distance < (inner_distance + outer_distance) / 2
            cell_coordinates = original_coordinates(coordinates, 5 / 16)
            cell_distance = periodic_distance_2d(cell_coordinates, center, 10)
        end

        #Set alpha according to cells position inside the circles.
        target_level = (cell_distance < inner_distance) + (cell_distance < outer_distance)
        alpha[element] = target_level / 2
    end
    return alpha
end

# For periodic domains, distance between two points must take into account
# periodic extensions of the domain
function periodic_distance_2d(coordinates, center, domain_length)
    dx = coordinates .- center
    dx_shifted = abs.(dx .% domain_length)
    dx_periodic = min.(dx_shifted, domain_length .- dx_shifted)
    return sqrt(sum(dx_periodic .^ 2))
end

#This takes a cells coordinates and transforms them into the coordinates of a parent-cell it originally refined from.
#It does it so that the parent-cell has given cell_length.
function original_coordinates(coordinates, cell_length)
    offset = coordinates .% cell_length
    offset_sign = sign.(offset)
    border = coordinates - offset
    center = border + (offset_sign .* cell_length / 2)
    return center
end

end # module TrixiExtension

import .TrixiExtension
###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_gauss

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-5.0, -5.0)
coordinates_max = (5.0, 5.0)

trees_per_dimension = (1, 1)

mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 4)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_controller = ControllerThreeLevel(semi,
                                      TrixiExtension.IndicatorSolutionIndependent(semi),
                                      base_level = 4,
                                      med_level = 5, med_threshold = 0.1,
                                      max_level = 6, max_threshold = 0.6)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 1.6)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback, stepsize_callback);

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
