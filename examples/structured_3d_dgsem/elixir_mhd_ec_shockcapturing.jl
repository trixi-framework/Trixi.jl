
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

equations = IdealGlmMhdEquations3D(1.4)

initial_condition = initial_condition_weak_blast_wave

surface_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
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

# Create a heavily warped curved mesh

# Mapping as described in https://arxiv.org/abs/2012.12040
function mapping(xi_, eta_, zeta_)
    # Transform input variables between -1 and 1 onto [0,3]
    xi = 1.5 * xi_ + 1.5
    eta = 1.5 * eta_ + 1.5
    zeta = 1.5 * zeta_ + 1.5

    y = eta +
        3 / 8 * (cos(1.5 * pi * (2 * xi - 3) / 3) *
         cos(0.5 * pi * (2 * eta - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    x = xi +
        3 / 8 * (cos(0.5 * pi * (2 * xi - 3) / 3) *
         cos(2 * pi * (2 * y - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    z = zeta +
        3 / 8 * (cos(0.5 * pi * (2 * x - 3) / 3) *
         cos(pi * (2 * y - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    return SVector(x, y, z)
end

cells_per_dimension = (8, 8, 8)
mesh = StructuredMesh(cells_per_dimension, mapping)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 1.4
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
