using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection diffusion equation

const a = 0.1
const nu = 0.1
const beta = 0.3

equations = LinearScalarAdvectionEquation1D(a)
equations_parabolic = LaplaceDiffusion1D(nu, equations)

dg = DGMulti(polydeg = 3, element_type = Line(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs))

initial_condition = function (x, t, equations::LinearScalarAdvectionEquation1D)
    return SVector(sin(x[1]))
end

source_terms = function (u, x, t, equations::LinearScalarAdvectionEquation1D)
    f = a * cos(x[1]) + nu * sin(x[1]) - beta * (cos(x[1])^2)
    return SVector(f)
end

source_terms_parabolic = function (u, gradients, x, t, equations::LaplaceDiffusion1D)
    dudx = gradients[1][1]
    return SVector(beta * dudx^2)
end

# Create a uniformly refined mesh with periodic boundaries
cells_per_dimension = (8,)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min = (-pi,), coordinates_max = (pi,),
                   periodicity = true)

boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, dg;
                                             source_terms = source_terms,
                                             source_terms_parabolic = source_terms_parabolic,
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))

alive_callback = AliveCallback(analysis_interval = 100)

cfl_advective = 0.5   # Not restrictive for this example
cfl_diffusive = 0.025 # Restricts the timestep
stepsize_callback = StepsizeCallback(cfl = cfl_advective,
                                     cfl_diffusive = cfl_diffusive)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL35(); adaptive = false, dt = stepsize_callback(ode),
            ode_default_options()..., callback = callbacks)
