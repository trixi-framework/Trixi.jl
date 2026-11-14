using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Free-stream preservation test on a `TreeMesh` with AMR.
# Since the solution is constant, a solution-based indicator would never trigger
# any adaptation. Instead, the elements are refined and coarsened randomly.

# define new structs inside a module to allow re-evaluating the file
module TrixiExtension

using Trixi
using Random: Xoshiro

# Indicator returning uniformly distributed random values in [0, 1) for each element,
# independent of the solution. The random number generator is seeded for reproducibility.
struct IndicatorRandom{RNG, Cache <: NamedTuple} <: Trixi.AbstractIndicator
    rng::RNG
    cache::Cache
end

function IndicatorRandom(semi; seed = 1234)
    basis = semi.solver.basis
    alpha = Vector{real(basis)}()
    cache = (; alpha)
    rng = Xoshiro(seed)
    return IndicatorRandom{typeof(rng), typeof(cache)}(rng, cache)
end

function (indicator::IndicatorRandom)(u, mesh, equations, dg, cache; kwargs...)
    alpha = indicator.cache.alpha
    resize!(alpha, nelements(dg, cache))
    for element in eachindex(alpha)
        alpha[element] = rand(indicator.rng, eltype(alpha))
    end
    return alpha
end

end # module TrixiExtension

import .TrixiExtension

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_constant

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
# The mesh is not refined beyond the initial uniform level during construction;
# all further refinement is done by the AMR callback.
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = BoundaryConditionDirichlet(initial_condition))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# Random refinement and coarsening between levels 2 and 4
amr_controller = ControllerThreeLevel(semi, TrixiExtension.IndicatorRandom(semi),
                                      base_level = 2,
                                      med_level = 3, med_threshold = 0.4,
                                      max_level = 4, max_threshold = 0.8)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           # The random indicator never settles, so the initial condition is
                           # not adapted; the mesh is refined during the simulation instead.
                           adapt_initial_condition = false)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
