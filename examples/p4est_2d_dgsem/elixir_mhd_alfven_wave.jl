
using OrdinaryDiffEq
using Trixi


###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
# equations = IdealGlmMhdEquations2D(1.4)

# initial_condition = initial_condition_rotor

# surface_flux = (flux_hll, flux_nonconservative_powell)
# volume_flux  = (flux_hindenlang_gassner, flux_nonconservative_powell)
# polydeg = 4
# basis = LobattoLegendreBasis(polydeg)
# indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                          alpha_max=0.5,
#                                          alpha_min=0.001,
#                                          alpha_smooth=true,
#                                          variable=density_pressure)
# volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                  volume_flux_dg=volume_flux,
#                                                  volume_flux_fv=surface_flux)
# solver = DGSEM(basis, surface_flux, volume_integral)

gamma = 5/3
equations = IdealGlmMhdEquations2D(gamma)

initial_condition = initial_condition_convergence_test

# Get the DG approximation space
volume_flux = (flux_central, flux_nonconservative_powell)
solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_powell),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

# Get the curved quad mesh from a mapping function
# Mapping as described in https://arxiv.org/abs/2012.12040
# function mapping_twist(xi, eta)
#   y = eta + 0.05 * (cos(1.5 * pi * xi) * sin(pi * eta))

#   x = xi + 0.05 * (sin(pi * xi) * cos(2 * pi * y))

#   return SVector(x, y)
# end

<<<<<<< Updated upstream
# Unstructured mesh with 16 cells of the square domain [0, 1]^2

coordinates_min = (0.0, 0.0)
=======
coordinates_min = (0.0      , 0.0      )
>>>>>>> Stashed changes
coordinates_max = (sqrt(2.0), sqrt(2.0))

trees_per_dimension = (8, 8)
mesh = P4estMesh(trees_per_dimension,
                 polydeg=3, initial_refinement_level=1,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 periodicity=true)

#mesh_file = joinpath(@__DIR__, "warpedmesh_16.inp")
# isfile(mesh_file) || download("https://gist.githubusercontent.com/efaulhaber/63ff2ea224409e55ee8423b3a33e316a/raw/7db58af7446d1479753ae718930741c47a3b79b7/square_unstructured_2.inp",
#                               mesh_file)

# mesh = P4estMesh{2}(mesh_file, polydeg=3,
#                     mapping=mapping_twist,
#                     initial_refinement_level=1)

#boundary_condition = BoundaryConditionDirichlet(initial_condition)
#boundary_conditions = Dict( :all => boundary_condition )

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)#,
 #                                   boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

# amr_indicator = IndicatorHennemannGassner(semi,
#                                           alpha_max=0.5,
#                                           alpha_min=0.001,
#                                           alpha_smooth=false,
#                                           variable=density_pressure)
# amr_controller = ControllerThreeLevel(semi, amr_indicator,
#                                       base_level=4,
#                                       max_level =6, max_threshold=0.01)
# amr_callback = AMRCallback(semi, amr_controller,
#                            interval=6,
#                            adapt_initial_condition=true,
#                            adapt_initial_condition_only_refine=true)

cfl = 0.35
stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        # amr_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
