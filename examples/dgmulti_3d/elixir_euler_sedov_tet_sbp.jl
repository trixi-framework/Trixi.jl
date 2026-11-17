using OrdinaryDiffEqSSPRK
# SummationByParts.jl provides the nodes of the SBP operators on tetrahedra. Loading it
# enables the corresponding package extension of StartUpDG.jl.
using SummationByParts: SummationByParts
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

"""
    initial_condition_medium_sedov_blast_wave(x, t, equations::CompressibleEulerEquations3D)

The Sedov blast wave setup based on example 35.1.4 from Flash
- https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
with smaller strength of the initial discontinuity.
"""
function initial_condition_medium_sedov_blast_wave(x, t,
                                                   equations::CompressibleEulerEquations3D)
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    z_norm = x[3] - inicenter[3]
    r = sqrt(x_norm^2 + y_norm^2 + z_norm^2)

    # TODO: Check radius
    # Setup based on example 35.1.4 in https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
    r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
    E = 1.0
    p0_inner = 3 * (equations.gamma - 1) * E / (4 * pi * r0^2)
    p0_outer = 1.0e-3

    # Calculate primitive variables
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_medium_sedov_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

# Shock capturing on tetrahedra with SBP nodes blends the flux differencing volume integral
# with the algebraic low-order method on the SBP nodes.
polydeg = 3
basis = DGMultiBasis(Tet(), polydeg, approximation_type = SBP())

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
dg = DGMulti(basis,
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = volume_integral)

cells_per_dimension = (8, 8, 8)
mesh = DGMultiMesh(dg, cells_per_dimension;
                   coordinates_min = (-1.0, -1.0, -1.0),
                   coordinates_max = (1.0, 1.0, 1.0),
                   periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.35)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode,
            SSPRK54(thread = Trixi.Threaded());
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
