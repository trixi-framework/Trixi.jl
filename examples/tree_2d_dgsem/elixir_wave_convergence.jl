using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the wave equations

equations = WaveEquations2D(2 * sqrt(2 / 5))

# initial condition for a standing wave
initial_condition = function (x, t, equations::WaveEquations2D)
    c = equations.c
    p = cospi(3*x[1] / 2) * cospi(x[2] / 2) * cospi(sqrt(5 / 2) * c * t)
    vx = 3 / sqrt(10) * sinpi(3 * x[1] / 2) * cospi(x[2] / 2) * sinpi(sqrt(5 / 2) * c * t)
    vy = 1 / sqrt(10) * cospi(3 * x[1] / 2) * sinpi(x[2] / 2) * sinpi(sqrt(5 / 2) * c * t)
    return SVector(p, vx, vy)
end

# corresponding boundary condition for the standing wave
boundary_condition = function (u_inner, orientation, direction, x, t,
                               surface_flux_function,
                               equations::WaveEquations2D)
    u_boundary = initial_condition(x, t, equations)
    # Calculate boundary flux
    if direction in (2, 4)  # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
    return flux
end

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000, periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false); dt = 1.0,
            ode_default_options()..., callback = callbacks);
