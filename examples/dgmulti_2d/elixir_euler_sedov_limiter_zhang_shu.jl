using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

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
    r0 = 0.5f0
    E = 1
    p0_inner = 3 * (equations.gamma - 1) * E / (3 * convert(RealT, pi) * r0^2)
    p0_outer = convert(RealT, 1.0e-5)

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

# Shock capturing on triangles with SBP nodes blends the flux differencing volume integral
# with the algebraic low-order method on the SBP nodes.
surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 3
basis = DGMultiBasis(Tri(), polydeg, approximation_type = SBP())
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
dg = DGMulti(basis,
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = volume_integral)

cells_per_dimension = (16, 16)
mesh = DGMultiMesh(dg, cells_per_dimension;
                   coordinates_min = (-2.0, -2.0), coordinates_max = (2.0, 2.0),
                   periodicity = true)

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

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.7)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        save_solution, stepsize_callback)

###############################################################################
# run the simulation

# Without the positivity-preserving limiter, the shock capturing alone does not keep the
# nodal values admissible and the simulation fails shortly after the start.
limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6),
                                               variables = (Trixi.density, pressure))

sol = solve(ode,
            CarpenterKennedy2N54(; stage_limiter! = limiter!,
                                 williamson_condition = false, thread = Trixi.Threaded());
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
