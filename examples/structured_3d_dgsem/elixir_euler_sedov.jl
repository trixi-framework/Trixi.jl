
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

#initial_condition = initial_condition_weak_blast_wave

###############################################################################
# Get the DG approximation space

initial_condition = initial_condition_sedov_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1.0,
                                         alpha_min=0.001,
                                         alpha_smooth=false,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
                                               
solver = DGSEM(polydeg=3, surface_flux=surface_flux, volume_integral=volume_integral)  

################################################################################
# Get the curved quad mesh from a file

# Mapping as described in https://arxiv.org/abs/2012.12040
#mapping(xi, eta, zeta) = SVector(xi, eta, zeta)

function mapping(xi, eta, zeta)
  y = eta + 0.125 * (cos(1.5 * pi * xi) * cos(0.5 * pi * eta) * cos(0.5 * pi * zeta))

  x = xi + 0.125 * (cos(0.5 * pi * xi) * cos(2 * pi * y) * cos(0.5 * pi * zeta))

  z = zeta + 0.125 * (cos(0.5 * pi * x) * cos(pi * y) * cos(0.5 * pi * zeta))

  return SVector(x, y, z)
end

cells_per_dimension = (4, 4, 4)

mesh = StructuredMesh(cells_per_dimension, mapping, periodicity=true)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 12.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=0.1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
