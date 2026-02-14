using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

The Sedov blast wave setup based on example 35.1.4 from Flash
- https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
"""
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations3D)
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(0, 0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    z_norm = x[3] - inicenter[3]
    r = sqrt(x_norm^2 + y_norm^2 + z_norm^2)

    # Setup based on example 35.1.4 in https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
    r0 = 0.21875f0 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
    E = 1
    nu = 3 # dims
    p0_inner = 3 * (equations.gamma - 1) * E / ((nu + 1) * convert(RealT, pi) * r0^nu)
    p0_outer = convert(RealT, 1.0e-5) # = true Sedov setup

    # Calculate primitive variables
    rho = 1
    v1 = 0
    v2 = 0
    v3 = 0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_sedov_blast_wave

basis = LobattoLegendreBasis(3)

# Use standard Hennemann-Gassner a-priori shock & blending indicator
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)

# In non-blended/limited regions, we use the cheaper weak form volume integral
volume_integral_default = VolumeIntegralWeakForm()

# For the blended/limited regions, we need to supply compatible high-order and low-order volume integrals.
volume_flux = flux_chandrashekar
volume_integral_blend_high_order = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral_blend_low_order = VolumeIntegralPureLGLFiniteVolume(volume_flux)

volume_integral = VolumeIntegralShockCapturingHGType(indicator_sc;
                                                    volume_integral_default = volume_integral_default,
                                                    volume_integral_blend_high_order = volume_integral_blend_high_order,
                                                    volume_integral_blend_low_order = volume_integral_blend_low_order)

volume_integral_stabilized = VolumeIntegralShockCapturingHG(indicator_sc;
                                                            volume_flux_dg = volume_flux,
                                                            volume_flux_fv = surface_flux)

volume_integral = VolumeIntegralAdaptive(volume_integral_default = VolumeIntegralWeakForm(),
                                         volume_integral_stabilized = volume_integral_stabilized,
                                         indicator = nothing) # Indicator taken from `volume_integral_stabilized`

surface_flux = flux_lax_friedrichs
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0, -2.0, -2.0)
coordinates_max = (2.0, 2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 1_000_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(alive_interval = 20)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 1.0,
                                          alpha_min = 0.0,
                                          alpha_smooth = false,
                                          variable = density_pressure)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 2,
                                      max_level = 6, max_threshold = 0.0003)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 2,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = false)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK54(thread = Trixi.True());
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
