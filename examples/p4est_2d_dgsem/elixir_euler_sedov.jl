using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

The Sedov blast wave setup based on example 35.1.4 from Flash
- https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
"""
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    # Setup based on example 35.1.4 in https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
    r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
    E = 1.0
    p0_inner = 3 * (equations.gamma - 1) * E / (3 * pi * r0^2)
    p0_outer = 1.0e-5 # = true Sedov setup

    # Calculate primitive variables
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(SVector(rho, v1, v2, p), equations)
end

initial_condition = initial_condition_sedov_blast_wave

# Get the DG approximation space

# `const flux_lax_friedrichs = flux_lax_friedrichs, i.e., `flux_lax_friedrichs`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `flux_lax_friedrichs`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the
# `StepsizeCallback` (CFL-Condition) and less diffusion.
surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

###############################################################################

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)

trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 4, initial_refinement_level = 2,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 12.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 300
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 300,
                                     save_initial_solution = true,
                                     save_final_solution = true)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
