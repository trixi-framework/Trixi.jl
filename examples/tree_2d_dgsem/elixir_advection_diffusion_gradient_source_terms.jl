using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection diffusion equation

const a = (0.1, 0.1)
const nu = 0.1
const beta = 0.3

equations = LinearScalarAdvectionEquation2D(a)
equations_parabolic = LaplaceDiffusion2D(nu, equations)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-Float64(pi), -Float64(pi))
coordinates_max = (Float64(pi), Float64(pi))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000,
                periodicity = true)

initial_condition = function (x, t, equations::LinearScalarAdvectionEquation2D)
    return SVector(sin(x[1]) * sin(x[2]))
end

source_terms = function (u, x, t, equations::LinearScalarAdvectionEquation2D)
    a1, a2 = a
    sinx, cosx = sincos(x[1])
    siny, cosy = sincos(x[2])

    f = a1 * cosx * siny + a2 * sinx * cosy +
        2 * nu * sinx * siny -
        beta * (cosx^2 * siny^2 + sinx^2 * cosy^2)
    return SVector(f)
end

source_terms_parabolic = function (u, gradients, x, t, equations)
    dudx = gradients[1][1]
    dudy = gradients[2][1]
    return SVector(beta * (dudx^2 + dudy^2))
end

# define periodic boundary conditions everywhere
boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition,
                                             solver;
                                             solver_parabolic = ViscousFormulationLocalDG(),
                                             source_terms = source_terms,
                                             source_terms_parabolic = source_terms_parabolic,
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval = 100)

cfl_advective = 0.5
cfl_diffusive = 0.01
stepsize_callback = StepsizeCallback(cfl = cfl_advective,
                                     cfl_diffusive = cfl_diffusive)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, RDPK3SpFSAL35(); adaptive = false, dt = stepsize_callback(ode),
            ode_default_options()..., callback = callbacks)
