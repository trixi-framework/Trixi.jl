using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection-diffusion equation

diffusivity() = 1.0e-2
advection_velocity = (-1.0, 1.0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)
equations_parabolic = LaplaceDiffusion2D(diffusivity(), equations)

function initial_condition_gauss_damped(x, t, equations)
    damping_factor = 1 + 4 * diffusivity() * t
    return SVector(exp(-(x[1]^2 + x[2]^2) / damping_factor) / damping_factor)
end
initial_condition = initial_condition_gauss_damped

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# This maps the domain [-1, 1]^2 to a 45-degree rotated increased square
square_size() = 5.0
function mapping(xi, eta)
    x = square_size() * xi
    y = square_size() * eta
    return SVector((x - y) / sqrt(2), (x + y) / sqrt(2))
end

trees_per_dimension = (23, 23)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 3, initial_refinement_level = 0,
                 mapping = mapping, periodicity = true)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = ViscousFormulationLocalDG(),
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

n_passes = 2
tspan = (0.0, n_passes * square_size() * sqrt(2))
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

time_int_tol = 1.0e-6
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
