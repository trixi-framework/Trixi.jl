using OrdinaryDiffEq
using Trixi

using DoubleFloats

###############################################################################
# semidiscretization of the linear advection equation

# See https://github.com/JuliaMath/DoubleFloats.jl
RealT = Double64

advection_velocity = 4 / 3 # Does not need to be in higher precision
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEM(RealT = RealT, polydeg = 7, surface_flux = flux_lax_friedrichs)

# CARE: Important to use higher precision datatype for coordinates
# as these are used for type promotion of the mesh (points etc.)
coordinates_min = -one(RealT) # minimum coordinate
coordinates_max = one(RealT) # maximum coordinate

# For `TreeMesh` the datatype has to be specified explicitly,
# i.e., is not inferred from the coordinates.
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000,
                RealT = RealT)

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
stepsize_callback = StepsizeCallback(cfl = 1.4)

callbacks = CallbackSet(summary_callback,
                        stepsize_callback,
                        analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, DP8(),
            # Turn off adaptivity to avoid setting very small tolerances
            adaptive = false,
            dt = 42, # `dt` does not need to be in higher precision
            ode_default_options()..., callback = callbacks);

# Print the timer summary
summary_callback()
