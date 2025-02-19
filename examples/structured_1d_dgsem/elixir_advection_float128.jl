using OrdinaryDiffEqFeagin
using Trixi

using Quadmath

###############################################################################
# semidiscretization of the linear advection equation

# See https://github.com/JuliaMath/Quadmath.jl
RealT = Float128

advection_velocity = 4 / 3 # Does not need to be in higher precision
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEM(RealT = RealT, polydeg = 13, surface_flux = flux_lax_friedrichs)

# CARE: Important to use higher precision datatype for coordinates
# as these are used for type promotion of the mesh (points etc.)
coordinates_min = (-one(RealT),)
coordinates_max = (one(RealT),)
cells_per_dimension = (1,)

# `StructuredMesh` infers datatype from coordinates
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

# CARE: Important to use higher precision datatype in specification of final time
tspan = (zero(RealT), one(RealT))

ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100,
                                     extra_analysis_errors = (:conservation_error,))

# cfl does not need to be in higher precision
stepsize_callback = StepsizeCallback(cfl = 0.25)

callbacks = CallbackSet(summary_callback,
                        stepsize_callback,
                        analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, Feagin14();
            # Turn off adaptivity to avoid setting very small tolerances
            adaptive = false,
            dt = 42, # `dt` does not need to be in higher precision
            ode_default_options()..., callback = callbacks);

# Print the timer summary
summary_callback()
