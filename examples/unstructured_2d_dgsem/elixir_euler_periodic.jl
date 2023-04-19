
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test
boundary_conditions = boundary_condition_periodic

###############################################################################
# Get the DG approximation space

solver = DGSEM(polydeg=6, surface_flux=FluxRotated(flux_hll))

###############################################################################
# Get the curved quad mesh from a file (downloads the file if not available locally)

default_mesh_file = joinpath(@__DIR__, "mesh_periodic_square_with_twist.mesh")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/12ce661d7c354c3d94c74b964b0f1c96/raw/8275b9a60c6e7ebbdea5fc4b4f091c47af3d5273/mesh_periodic_square_with_twist.mesh",
                                       default_mesh_file)
mesh_file = default_mesh_file

mesh = UnstructuredMesh2D(mesh_file, periodicity=true)

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

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(); ode_default_options()..., callback=callbacks);
summary_callback() # print the timer summary
