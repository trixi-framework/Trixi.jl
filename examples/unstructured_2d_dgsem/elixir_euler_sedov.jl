
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_medium_sedov_blast_wave

# Get the DG approximation space
surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 6
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1.0,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux, volume_integral=volume_integral)

# Get the curved quad mesh from a file
default_mesh_file = joinpath(@__DIR__, "mesh_periodic_square_with_twist.mesh")

mesh_file = default_mesh_file

mesh = UnstructuredMesh2D(mesh_file, periodicity=true)

# create the semidiscretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 12.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 300
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=300,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=0.5)

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
