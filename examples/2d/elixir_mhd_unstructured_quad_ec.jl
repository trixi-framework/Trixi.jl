
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

equations = IdealGlmMhdEquations2D(5/3)

initial_condition = initial_condition_weak_blast_wave

###############################################################################
# Get the DG approximation space

volume_flux = flux_derigs_etal
solver = DGSEM(polydeg=6, surface_flux=FluxRotated(flux_derigs_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the curved quad mesh from a file

default_mesh_file = joinpath(@__DIR__, "mesh_periodic_square_with_twist.mesh")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/12ce661d7c354c3d94c74b964b0f1c96/raw/8275b9a60c6e7ebbdea5fc4b4f091c47af3d5273/mesh_periodic_square_with_twist.mesh",
                                       default_mesh_file)
mesh_file = default_mesh_file

mesh = UnstructuredQuadMesh(mesh_file, periodicity=true)

###############################################################################
# Get the curved quad mesh from a mapping function
# TODO: use this (more difficult) twisted mesh for the EC test once the CurvedMesh{2} is available for MHD
#
# # Mapping as described in https://arxiv.org/abs/2012.12040, but reduced to 2D
# function mapping(xi_, eta_)
#   # Transform input variables between -1 and 1 onto [0,3]
#   xi = 1.5 * xi_ + 1.5
#   eta = 1.5 * eta_ + 1.5
#
#   y = eta + 3/8 * (cos(1.5 * pi * (2 * xi - 3)/3) *
#                    cos(0.5 * pi * (2 * eta - 3)/3))
#
#   x = xi + 3/8 * (cos(0.5 * pi * (2 * xi - 3)/3) *
#                   cos(2 * pi * (2 * y - 3)/3))
#
#   return SVector(x, y)
# end
#
# cells_per_dimension = (8, 8)
#
# # Create curved mesh with 8 x 8 elements
# mesh = CurvedMesh(cells_per_dimension, mapping)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                     extra_analysis_integrals=(entropy, energy_total,
                                                               energy_kinetic, energy_internal,
                                                               energy_magnetic, cross_helicity))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)
cfl = 1.0
stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
