
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_constant
boundary_conditions = Dict( "Body"    => initial_condition,
                            "Button1" => initial_condition,
                            "Button2" => initial_condition,
                            "Eye1"    => initial_condition,
                            "Eye2"    => initial_condition,
                            "Smile"   => initial_condition,
                            "Bowtie"  => initial_condition )

###############################################################################
# Get the DG approximation space

solver = DGSEM(polydeg=6, surface_flux=flux_hll)

###############################################################################
# Get the curved quad mesh from a file

mesh_file = joinpath(@__DIR__, "mesh_gingerbread_man.mesh")

mesh = UnstructuredQuadMesh(mesh_file)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
