
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

equations = IdealGlmMhdEquations3D(1.4)

"""
    initial_condition_blast_wave(x, t, equations::IdealGlmMhdEquations3D)

Weak magnetic blast wave setup taken from Section 6.1 of the paper:
- A. M. Rueda-Ramírez, S. Hennemann, F. J. Hindenlang, A. R. Winters, G. J. Gassner (2021)
  An entropy stable nodal discontinuous Galerkin method for the resistive MHD
  equations. Part II: Subcell finite volume shock capturing
  [doi: 10.1016/j.jcp.2021.110580](https://doi.org/10.1016/j.jcp.2021.110580)
"""
function initial_condition_blast_wave(x, t, equations::IdealGlmMhdEquations3D)
    # Center of the blast wave is selected for the domain [0, 3]^3
    inicenter = (1.5, 1.5, 1.5)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    z_norm = x[3] - inicenter[3]
    r = sqrt(x_norm^2 + y_norm^2 + z_norm^2)

    delta_0 = 0.1
    r_0 = 0.3
    lambda = exp(5.0 / delta_0 * (r - r_0))

    prim_inner = SVector(1.2, 0.1, 0.0, 0.1, 0.9, 1.0, 1.0, 1.0, 0.0)
    prim_outer = SVector(1.2, 0.2, -0.4, 0.2, 0.3, 1.0, 1.0, 1.0, 0.0)
    prim_vars = (prim_inner + lambda * prim_outer) / (1.0 + lambda)

    return prim2cons(prim_vars, equations)
end
initial_condition = initial_condition_blast_wave

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
polydeg = 3
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

# Mapping as described in https://arxiv.org/abs/2012.12040 but with slightly less warping.
# The mapping will be interpolated at tree level, and then refined without changing
# the geometry interpolant.
function mapping(xi_, eta_, zeta_)
    # Transform input variables between -1 and 1 onto [0,3]
    xi = 1.5 * xi_ + 1.5
    eta = 1.5 * eta_ + 1.5
    zeta = 1.5 * zeta_ + 1.5

    y = eta +
        3 / 11 * (cos(1.5 * pi * (2 * xi - 3) / 3) *
         cos(0.5 * pi * (2 * eta - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    x = xi +
        3 / 11 * (cos(0.5 * pi * (2 * xi - 3) / 3) *
         cos(2 * pi * (2 * y - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    z = zeta +
        3 / 11 * (cos(0.5 * pi * (2 * x - 3) / 3) *
         cos(pi * (2 * y - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    return SVector(x, y, z)
end

trees_per_dimension = (2, 2, 2)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 3,
                 mapping = mapping,
                 initial_refinement_level = 2,
                 periodicity = true)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

amr_indicator = IndicatorLöhner(semi,
                                variable = density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 2,
                                      max_level = 4, max_threshold = 0.15)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

cfl = 1.4
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        amr_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
