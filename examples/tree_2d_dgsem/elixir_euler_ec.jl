
using OrdinaryDiffEq
using Trixi
using Random: seed!
###############################################################################
# semidiscretization of the compressible Euler equations
equations = CompressibleEulerEquations2D(1.4)

seed!(1)
function initial_condition_random_field(x, t, equations::CompressibleEulerEquations2D)
amplitude = 1.5
rho = 2 + amplitude * rand() 
v1 = -3.1 + amplitude * rand()
v2 = 1.3 + amplitude * rand() 
p = 7.54 + amplitude * rand()
return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_weak_blast_wave
# initial_condition = initial_condition_random_field

#volume_flux = flux_ranocha
#solver = DGSEM(polydeg = 3, surface_flux = flux_ranocha,
#               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
#
surface_flux = flux_ranocha
polydeg = 3
basis = GaussLegendreBasis(polydeg; polydeg_projection = 1 * polydeg, polydeg_cutoff = 3)
volume_integral = VolumeIntegralWeakFormProjection()
#volume_integral = VolumeIntegralWeakForm()
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 10_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition_periodic)

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

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        #save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
