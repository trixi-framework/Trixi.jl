
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

function initial_condition_weak_blast_wave(x, t,
                                           equations::CompressibleEulerEquations3D)
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    z_norm = x[3] - inicenter[3]
    r = sqrt(x_norm^2 + y_norm^2 + z_norm^2)

    r0 = 0.2
    E = 1.0
    p0_inner = 3
    p0_outer = 1

    # Calculate primitive variables
    rho = 1.1
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

initial_condition = initial_condition_weak_blast_wave

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

# Setup a periodic mesh with 4 x 4 x 4 trees and 8 x 8 x 8 elements
trees_per_dimension = (4, 4, 4)

# Affine type mapping to take the [-1,1]^3 domain
# and warp it as described in https://arxiv.org/abs/2012.12040
function mapping_twist(xi, eta, zeta)
    y = eta + 1 / 6 * (cos(1.5 * pi * xi) * cos(0.5 * pi * eta) * cos(0.5 * pi * zeta))

    x = xi + 1 / 6 * (cos(0.5 * pi * xi) * cos(2 * pi * y) * cos(0.5 * pi * zeta))

    z = zeta + 1 / 6 * (cos(0.5 * pi * x) * cos(pi * y) * cos(0.5 * pi * zeta))

    return SVector(x, y, z)
end

mesh = P4estMesh(trees_per_dimension,
                 polydeg = 2,
                 mapping = mapping_twist,
                 initial_refinement_level = 1,
                 periodicity = true)

# Create the semidiscretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

amr_indicator = IndicatorLöhner(semi, variable = Trixi.density)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 1,
                                      med_level = 2, med_threshold = 0.05,
                                      max_level = 3, max_threshold = 0.15)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 1,
                           adapt_initial_condition = false,
                           adapt_initial_condition_only_refine = false)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
