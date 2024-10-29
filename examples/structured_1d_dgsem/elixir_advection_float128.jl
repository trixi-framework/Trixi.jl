
using OrdinaryDiffEq
using Trixi

using Quadmath

###############################################################################
# semidiscretization of the linear advection equation

# See https://github.com/JuliaMath/Quadmath.jl
RealT = Float128

advection_velocity = 4 / 3 # Does not need to be in higher precision
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEM(RealT = RealT, polydeg = 7, surface_flux = flux_lax_friedrichs)

# CARE: Important to use higher precision datatype for coordinates
# as these are used for type promotion of the mesh (points etc.)
coordinates_min = (-RealT(1),) # minimum coordinate
coordinates_max = (RealT(1),) # maximum coordinate
cells_per_dimension = (256,)

# NOTE: StructuredMesh supports higher precision coordinates
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

# CARE: Important to use higher precision datatype in specification of final time
tspan = (zero(RealT), RealT(1) / 1000)

ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 1000,
                                     extra_analysis_errors = (:conservation_error,))

# cfl does not need to be in higher precision
stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
                        stepsize_callback,
                        analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(),
            dt = 42.0, # `dt` does not need to be in higher precision
            save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
