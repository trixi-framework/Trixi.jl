# Linear advection-diffusion with source terms depending on the gradient on a periodic
# curved mesh with the continuous Galerkin spectral element method (CGSEM), cf.
# examples/tree_2d_dgsem/elixir_advection_diffusion_gradient_source_terms.jl

using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection-diffusion equation with the source term
# `beta * |grad(u)|^2` in the parabolic part

const a = (0.1, 0.1)
const nu = 0.1
const beta = 0.3

equations = LinearScalarAdvectionEquation2D(a)
equations_parabolic = LaplaceDiffusion2D(nu, equations)

# Create CGSEM solver with polynomial degree = 3. The surface flux enters only through
# weakly imposed boundary conditions, so it is unused on this periodic mesh.
solver = CGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# This maps the domain [-1, 1]^2 to [-pi, pi]^2 while also
# introducing a curved warping to interior nodes.
function mapping(xi, eta)
    x = xi + 0.1 * sin(pi * xi) * sin(pi * eta)
    y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
    return pi * SVector(x, y)
end

trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 3, initial_refinement_level = 1,
                 mapping = mapping,
                 periodicity = true)

# The source terms are chosen such that u(x, y, t) = sin(x) * sin(y) is a steady
# solution of
#   u_t + a ⋅ ∇u = nu * Δu + beta * |∇u|^2 + f(x, y).
initial_condition = function (x, t, equations::LinearScalarAdvectionEquation2D)
    return SVector(sin(x[1]) * sin(x[2]))
end

# Hyperbolic source terms f(x, y)
source_terms = function (u, x, t, equations::LinearScalarAdvectionEquation2D)
    a1, a2 = a
    sinx, cosx = sincos(x[1])
    siny, cosy = sincos(x[2])
    f = a1 * cosx * siny + a2 * sinx * cosy +
        2 * nu * sinx * siny -
        beta * (cosx^2 * siny^2 + sinx^2 * cosy^2)
    return SVector(f)
end

# Parabolic source terms depending on the gradient. CGSEM evaluates them with the
# element-local gradients and adds them before the assembly.
source_terms_parabolic = function (u, gradients, x, t, equations)
    dudx = gradients[1][1]
    dudy = gradients[2][1]
    return SVector(beta * (dudx^2 + dudy^2))
end

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition,
                                             solver;
                                             source_terms = source_terms,
                                             source_terms_parabolic = source_terms_parabolic,
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval = analysis_interval)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
time_int_tol = 1.0e-10
sol = solve(ode, RDPK3SpFSAL49(thread = Trixi.Threaded());
            abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
