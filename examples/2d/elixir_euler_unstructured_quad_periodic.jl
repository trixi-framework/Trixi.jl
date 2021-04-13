
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

polydeg = 6
surface_flux = flux_hll
solver = DGSEM(polydeg, surface_flux)

###############################################################################
# Get the curved quad mesh from a file

mesh_file = joinpath(@__DIR__, "PeriodicXandY20.mesh")
periodicity = true
mesh = UnstructuredQuadMesh(Float64, mesh_file, periodicity)

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

sol = solve(ode, SSPRK43(), save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
