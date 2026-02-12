using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the invsicid Burgers' equation with diffusion

equations = InviscidBurgersEquation1D()
diffusivity() = 1e-2
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

coordinates_min = -0.25
coordinates_max = 0.75

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000,
                periodicity = false)

# This initial condition is a simplification/analogy to the
# Navier Stokes viscous shock (Becker-Morduchow-Libby solution).
function initial_condition_weak_shock_wave(x, t, equations)
    x_shock = -0.0
    x_translated = (x[1] - x_shock) - t # shock speed is 1
    u = 2 / (1 + exp(x_translated / diffusivity()))

    return SVector(u)
end
initial_condition = initial_condition_weak_shock_wave

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = boundary_condition
boundary_conditions_parabolic = boundary_condition

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100)

alive_callback = AliveCallback(analysis_interval = 100)

# Timestep is limited by diffusive CFL
stepsize_callback = StepsizeCallback(cfl = 0.8, cfl_diffusive = 0.15)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3Sp510();
            adaptive = false, dt = 1.0,
            ode_default_options()..., callback = callbacks);
