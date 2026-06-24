using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

A slight modification of the Sedov blast wave setup based on example 35.1.4 from Flash
- https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
"""
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    # Setup based on example 35.1.4 in https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
    r0 = 0.5
    E = 1
    p0_inner = 3 * (equations.gamma - 1) * E / (3 * convert(RealT, pi) * r0^2)

    # slightly smaller initial pressure to trigger the Liu-Zhang limiter
    p0_outer = convert(RealT, 1.0e-6)

    # Calculate primitive variables
    rho = 1
    v1 = 0
    v2 = 0
    p = r > r0 ? p0_outer : p0_inner
    if r ≈ r0
        p = 0.5f0 * (p0_inner + p0_outer)
    end

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_sedov_blast_wave

surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(3)
indicator_ec = IndicatorEntropyCorrection(equations, basis)
volume_integral_default = VolumeIntegralWeakForm()
volume_integral_entropy_stable = VolumeIntegralPureLGLFiniteVolume(surface_flux)
volume_integral = VolumeIntegralAdaptive(indicator_ec,
                                         volume_integral_default,
                                         volume_integral_entropy_stable)

dg = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 100_000, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.8)

###############################################################################
# run the simulation

callbacks = CallbackSet(summary_callback, analysis_callback,
                        alive_callback, stepsize_callback)

local_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1e-8, 1e-8),
                                                     variables = (Trixi.density,
                                                                  energy_internal))
global_limiter! = PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
                                                      record_davis_yin_iterations = true)

ode_solver = CarpenterKennedy2N54(; stage_limiter! = global_limiter!,
                                  step_limiter! = global_limiter!,
                                  williamson_condition = false)

sol = solve(ode, ode_solver;
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
