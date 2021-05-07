
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict( "Bottom" => boundary_condition_convergence_test,
                            "Top"    => boundary_condition_convergence_test,
                            "Right"  => boundary_condition_convergence_test,
                            "Left"   => boundary_condition_convergence_test,
                            "Circle" => boundary_condition_convergence_test )

###############################################################################
# Get the DG approximation space

solver = DGSEM(polydeg=5, surface_flux=FluxRotated(flux_hll))

###############################################################################
# Get the curved quad mesh from a file (downloads the file if not available locally)

default_mesh_file = joinpath(@__DIR__, "mesh_box_around_circle.mesh")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/8b9b11a1eedfa54b215c122c3d17b271/raw/0d2b5d98c87e67a6f384693a8b8e54b4c9fcbf3d/mesh_box_around_circle.mesh",
                                       default_mesh_file)
mesh_file = default_mesh_file

mesh = UnstructuredQuadMesh(mesh_file)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=3.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
