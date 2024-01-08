using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations.

equations = ShallowWaterEquations2D(gravity_constant = 9.81)

initial_condition = initial_condition_convergence_test # MMS EOC test

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg = 3,
               surface_flux = (flux_lax_friedrichs, flux_nonconservative_fjordholm_etal),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the P4estMesh and setup a periodic mesh

coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (sqrt(2.0), sqrt(2.0))  # maximum coordinates (max(x), max(y))

mapping = Trixi.coordinates2mapping(coordinates_min, coordinates_max)

trees_per_dimension = (8, 8)

mesh = T8codeMesh(trees_per_dimension, polydeg = 3,
                  mapping = mapping,
                  initial_refinement_level = 1)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-8, reltol = 1.0e-8,
            ode_default_options()..., callback = callbacks);
summary_callback() # print the timer summary
