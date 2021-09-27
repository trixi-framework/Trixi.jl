
using OrdinaryDiffEq
using Trixi

# define new structs inside a module to allow re-evaluating the file
module TrixiExtension

using Trixi

struct IndicatorVortex{Cache<:NamedTuple} <: Trixi.AbstractIndicator
  cache::Cache
end

function IndicatorVortex(semi)
  basis = semi.solver.basis
  alpha = Vector{real(basis)}()
  A = Array{real(basis), 2}
  indicator_threaded = [A(undef, nnodes(basis), nnodes(basis))
                        for _ in 1:Threads.nthreads()]
  cache = (; semi.mesh, alpha, indicator_threaded)

  return IndicatorVortex{typeof(cache)}(cache)
end

function (indicator_vortex::IndicatorVortex)(u::AbstractArray{<:Any,4},
                                             mesh, equations, dg, cache;
                                             t, kwargs...)
  mesh = indicator_vortex.cache.mesh
  alpha = indicator_vortex.cache.alpha
  indicator_threaded = indicator_vortex.cache.indicator_threaded
  resize!(alpha, nelements(dg, cache))


  # get analytical vortex center (based on assumption that center=[0.0,0.0]
  # at t=0.0 and that we stop after one period)
  domain_length = mesh.tree.length_level_0
  if t < 0.5 * domain_length
    center = (t, t)
  else
    center = (t-domain_length, t-domain_length)
  end

  Threads.@threads for element in eachelement(dg, cache)
    cell_id = cache.elements.cell_ids[element]
    coordinates = (mesh.tree.coordinates[1, cell_id], mesh.tree.coordinates[2, cell_id])
    # use the negative radius as indicator since the AMR controller increases
    # the level with increasing value of the indicator and we want to use
    # high levels near the vortex center
    alpha[element] = -periodic_distance_2d(coordinates, center, domain_length)
  end

  return alpha
end

function periodic_distance_2d(coordinates, center, domain_length)
  dx = @. abs(coordinates - center)
  dx_periodic = @. min(dx, domain_length - dx)
  return sqrt(sum(abs2, dx_periodic))
end

end # module TrixiExtension

import .TrixiExtension

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_isentropic_vortex
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-10, -10)
coordinates_max = ( 10,  10)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200

analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     extra_analysis_errors=(:conservation_error,),
                                     extra_analysis_integrals=(entropy, energy_total,
                                                               energy_kinetic, energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=50,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

amr_controller = ControllerThreeLevel(semi, TrixiExtension.IndicatorVortex(semi),
                                      base_level=3,
                                      med_level=4, med_threshold=-3.0,
                                      max_level=5, max_threshold=-2.0)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=1.1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback, stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
