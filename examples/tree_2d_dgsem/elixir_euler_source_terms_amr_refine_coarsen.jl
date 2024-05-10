# This elixir and indicator is only for testing purposes and does not have any practical use

using OrdinaryDiffEq
using Trixi

# Define new structs inside a module to allow re-evaluating the file.
# This module name needs to be unique among all examples, otherwise Julia will throw warnings
# if multiple test cases using the same module name are run in the same session.
module TrixiExtensionEulerAMR

using Trixi

struct IndicatorRefineCoarsen{Cache <: NamedTuple} <: Trixi.AbstractIndicator
    cache::Cache
end

function IndicatorRefineCoarsen(semi)
    basis = semi.solver.basis
    alpha = Vector{real(basis)}()
    cache = (; semi.mesh, alpha)

    return IndicatorRefineCoarsen{typeof(cache)}(cache)
end

function (indicator::IndicatorRefineCoarsen)(u::AbstractArray{<:Any, 4},
                                             mesh, equations, dg, cache;
                                             t, kwargs...)
    alpha = indicator.cache.alpha
    resize!(alpha, nelements(dg, cache))

    if t >= 0.7 && t < 1.0
        # Refine to max level
        alpha .= 1.0
    elseif t >= 1.0
        # Coarsen to base level
        alpha .= -1.0
    else
        alpha .= 0.0
    end

    return alpha
end

end # module TrixiExtensionEulerAMR

import .TrixiExtensionEulerAMR

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (0.0, 0.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_controller = ControllerThreeLevel(semi,
                                      TrixiExtensionEulerAMR.IndicatorRefineCoarsen(semi),
                                      base_level = 3, max_level = 6,
                                      med_threshold = 0.1, max_threshold = 0.6)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution, amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
