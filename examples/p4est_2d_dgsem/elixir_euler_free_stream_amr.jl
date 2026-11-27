using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Free-stream preservation test on a curved `P4estMesh` with AMR.
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

# Mapping as described in https://arxiv.org/abs/2012.12040 but reduced to 2D
function mapping(xi_, eta_)
    # Transform input variables between -1 and 1 onto [0,3]
    xi = 1.5 * xi_ + 1.5
    eta = 1.5 * eta_ + 1.5

    y = eta + 3 / 8 * (cos(1.5 * pi * (2 * xi - 3) / 3) *
                       cos(0.5 * pi * (2 * eta - 3) / 3))

    x = xi + 3 / 8 * (cos(0.5 * pi * (2 * xi - 3) / 3) *
                      cos(2 * pi * (2 * y - 3) / 3))

    return SVector(x, y)
end

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

# Unstructured mesh with 48 cells of the square domain [-1, 1]^n
mesh_file = Trixi.download("https://gist.githubusercontent.com/efaulhaber/a075f8ec39a67fa9fad8f6f84342cbca/raw/a7206a02ed3a5d3cadacd8d9694ac154f9151db7/square_unstructured_1.inp",
                           joinpath(@__DIR__, "square_unstructured_1.inp"))

# Map the unstructured mesh with the mapping above.
# The mesh is not refined during construction; all refinement is done by the AMR callback.
mesh = P4estMesh{2}(mesh_file, polydeg = 3, mapping = mapping, initial_refinement_level = 0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = (;
                                                           all = BoundaryConditionDirichlet(initial_condition)))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# Random refinement and coarsening between levels 0 and 2
amr_controller = ControllerThreeLevel(semi, TrixiExtension.IndicatorRandom(semi),
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.4,
                                      max_level = 2, max_threshold = 0.8)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           # The random indicator never settles, so the initial condition is
                           # not adapted; the mesh is refined during the simulation instead.
                           adapt_initial_condition = false)

stepsize_callback = StepsizeCallback(cfl = 2.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
