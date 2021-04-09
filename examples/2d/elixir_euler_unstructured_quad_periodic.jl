
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)
num_eqns = nvariables(equations)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test
boundary_conditions = boundary_condition_periodic

###############################################################################
# Get the DG approximation space

poly_deg = 6
surface_flux = flux_hll # flux_lax_friedrichs
solver = DGSEM(poly_deg, surface_flux)

###############################################################################
# Get the curved quad mesh from a file

mesh_file = "./examples/2d/PeriodicXandY20.mesh"
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

limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-5, 5.0e-5),
                                               variables=(Trixi.density, pressure))
stage_limiter! = limiter!
step_limiter!  = limiter!

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(stage_limiter!, step_limiter!), save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
