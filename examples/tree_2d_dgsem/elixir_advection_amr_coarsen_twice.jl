# This elixir and indicator is only for testing purposes and does not have any practical use

using OrdinaryDiffEq
using Trixi

# Define new structs inside a module to allow re-evaluating the file.
# This module name needs to be unique among all examples, otherwise Julia will throw warnings
# if multiple test cases using the same module name are run in the same session.
module TrixiExtensionCoarsen

using Trixi

struct IndicatorAlwaysCoarsen{Cache <: NamedTuple} <: Trixi.AbstractIndicator
    cache::Cache
end

function IndicatorAlwaysCoarsen(semi)
    basis = semi.solver.basis
    alpha = Vector{real(basis)}()
    cache = (; semi.mesh, alpha)

    return IndicatorAlwaysCoarsen{typeof(cache)}(cache)
end

function (indicator::IndicatorAlwaysCoarsen)(u::AbstractArray{<:Any, 4},
                                             mesh, equations, dg, cache;
                                             t, kwargs...)
    alpha = indicator.cache.alpha
    resize!(alpha, nelements(dg, cache))

    alpha .= -1.0

    return alpha
end

end # module TrixiExtensionCoarsen

import .TrixiExtensionCoarsen

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_gauss

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-5.0, -5.0)
coordinates_max = (5.0, 5.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
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
                                      TrixiExtensionCoarsen.IndicatorAlwaysCoarsen(semi),
                                      base_level = 2, max_level = 2,
                                      med_threshold = 0.1, max_threshold = 0.6)

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
