
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_constant

surface_flux = FluxRotated(flux_lax_friedrichs)
volume_integral = VolumeIntegralWeakForm()
solver = DGSEM(3, surface_flux, volume_integral)

# Mapping described in https://arxiv.org/abs/2012.12040,
# but on [-1,1]^3 instead of [0,3]^3
function mapping(xi, eta, zeta)
  y = eta + 0.25 * cos(1.5 * pi * xi) * cos(0.5 * pi * eta) * cos(0.5 * pi * zeta)

  x = xi + 0.25 * cos(0.5 * pi * xi) * cos(2 * pi * y) * cos(0.5 * pi * zeta)

  z = zeta + 0.25 * cos(0.5 * pi * x) * cos(pi * y) * cos(0.5 * pi * zeta)
  
  return SVector(x, y, z)
end

cells_per_dimension = (4, 4, 4)

mesh = CurvedMesh(cells_per_dimension, mapping)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart, save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
