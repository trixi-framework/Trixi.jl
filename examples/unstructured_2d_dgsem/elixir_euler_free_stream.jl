
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_constant

boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:Body => boundary_condition_free_stream,
                           :Button1 => boundary_condition_free_stream,
                           :Button2 => boundary_condition_free_stream,
                           :Eye1 => boundary_condition_free_stream,
                           :Eye2 => boundary_condition_free_stream,
                           :Smile => boundary_condition_free_stream,
                           :Bowtie => boundary_condition_free_stream)

###############################################################################
# Get the DG approximation space

solver = DGSEM(polydeg = 6, surface_flux = flux_hll)

###############################################################################
# Get the curved quad mesh from a file (downloads the file if not available locally)
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/2c6440b5f8a57db131061ad7aa78ee2b/raw/1f89fdf2c874ff678c78afb6fe8dc784bdfd421f/mesh_gingerbread_man.mesh",
                           joinpath(@__DIR__, "mesh_gingerbread_man.mesh"))

mesh = UnstructuredMesh2D(mesh_file)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
