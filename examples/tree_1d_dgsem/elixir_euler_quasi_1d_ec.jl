using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the quasi 1d compressible Euler equations with a discontinuous nozzle width function.
# See Chan et al.  https://doi.org/10.48550/arXiv.2307.12089 for details

equations = CompressibleEulerEquationsQuasi1D(1.4)

# Setup a truly discontinuous density function and nozzle width for
# this academic testcase of entropy conservation. The errors from the analysis
# callback are not important but the entropy error for this test case
# `∑∂S/∂U ⋅ Uₜ` should be around machine roundoff.
# Works as intended for TreeMesh1D with `initial_refinement_level=6`. If the mesh
# refinement level is changed the initial condition below may need changed as well to
# ensure that the discontinuities lie on an element interface.
function initial_condition_ec(x, t, equations::CompressibleEulerEquationsQuasi1D)
    v1 = 0.1
    rho = 2.0 + 0.1 * x[1]
    p = 3.0
    a = 2.0 + x[1]

    return prim2cons(SVector(rho, v1, p, a), equations)
end

initial_condition = initial_condition_ec

surface_flux = (flux_chan_etal, flux_nonconservative_chan_etal)
volume_flux = surface_flux
solver = DGSEM(polydeg = 4, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0,)
coordinates_max = (1.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
