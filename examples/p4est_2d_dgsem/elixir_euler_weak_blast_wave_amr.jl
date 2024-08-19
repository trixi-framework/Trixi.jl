
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

function initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    r0 = 0.2
    E = 1
    p0_inner = 3
    p0_outer = 1

    # Calculate primitive variables
    rho = 1.1
    v1 = 0.0
    v2 = 0.0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(SVector(rho, v1, v2, p), equations)
end

initial_condition = initial_condition_weak_blast_wave

# Get the DG approximation space

# Activate the shock capturing + flux differencing
surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

###############################################################################

# Affine type mapping to take the [-1,1]^2 domain
# and warp it as described in https://arxiv.org/abs/2012.12040
# Warping with the coefficient 0.2 is even more extreme.
function mapping_twist(xi, eta)
    y = eta + 0.125 * cos(1.5 * pi * xi) * cos(0.5 * pi * eta)
    x = xi + 0.125 * cos(0.5 * pi * xi) * cos(2.0 * pi * y)
    return SVector(x, y)
end

# The mesh below can be made periodic
# Create P4estMesh with 8 x 8 trees
trees_per_dimension = (8, 8)
mesh = P4estMesh(trees_per_dimension, polydeg = 4,
                 mapping = mapping_twist,
                 initial_refinement_level = 0,
                 periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 400
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 0.2,
                                     save_initial_solution = true,
                                     save_final_solution = true)

amr_indicator = IndicatorLöhner(semi, variable = Trixi.density)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.05,
                                      max_level = 2, max_threshold = 0.1)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);#, maxiters=4);
summary_callback() # print the timer summary
