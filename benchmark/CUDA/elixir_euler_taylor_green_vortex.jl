using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

function initial_condition_taylor_green_vortex(x, t,
                                               equations::CompressibleEulerEquations3D)
    A = 1.0 # magnitude of speed
    Ms = 0.1 # maximum Mach number

    rho = 1.0
    v1 = A * sin(x[1]) * cos(x[2]) * cos(x[3])
    v2 = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
    v3 = 0.0
    p = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
    p = p +
        1.0 / 16.0 * A^2 * rho *
        (cos(2 * x[1]) * cos(2 * x[3]) +
         2 * cos(2 * x[2]) + 2 * cos(2 * x[1]) + cos(2 * x[2]) * cos(2 * x[3]))

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

initial_condition = initial_condition_taylor_green_vortex

# TODO Undefined external symbol "log"
#volume_flux = flux_ranocha
volume_flux = flux_lax_friedrichs
solver = DGSEM(polydeg = 5, surface_flux = volume_flux)
# TODO flux diff
#volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0, -1.0) .* pi
coordinates_max = (1.0, 1.0, 1.0) .* pi

initial_refinement_level = 1
trees_per_dimension = (4, 4, 4)

mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = true, initial_refinement_level = initial_refinement_level)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 100.0)
ode = semidiscretize(semi, tspan; storage_type = nothing, real_type = nothing)

summary_callback = SummaryCallback()

stepsize_callback = StepsizeCallback(cfl = 0.1)

callbacks = CallbackSet(summary_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

maxiters = 200
run_profiler = false

# disable warnings when maxiters is reached
integrator = init(ode, CarpenterKennedy2N54(williamson_condition = false),
                  dt = 1.0,
                  save_everystep = false, callback = callbacks,
                  maxiters = maxiters, verbose = false)
if run_profiler
    prof_result = CUDA.@profile solve!(integrator)
else
    solve!(integrator)
    prof_result = nothing
end

finalize(mesh)
