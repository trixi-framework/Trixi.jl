using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
equations = IdealGlmMhdEquations2D(1.4)

"""
    initial_condition_rotor(x, t, equations::IdealGlmMhdEquations2D)

The classical MHD rotor test case. Here, the setup is taken from
- Dominik Derigs, Gregor J. Gassner, Stefanie Walch & Andrew R. Winters (2018)
  Entropy Stable Finite Volume Approximations for Ideal Magnetohydrodynamics
  [doi: 10.1365/s13291-018-0178-9](https://doi.org/10.1365/s13291-018-0178-9)
"""
function initial_condition_rotor(x, t, equations::IdealGlmMhdEquations2D)
    # setup taken from Derigs et al. DMV article (2018)
    # domain must be [0, 1] x [0, 1], γ = 1.4
    dx = x[1] - 0.5
    dy = x[2] - 0.5
    r = sqrt(dx^2 + dy^2)
    f = (0.115 - r) / 0.015
    if r <= 0.1
        rho = 10.0
        v1 = -20.0 * dy
        v2 = 20.0 * dx
    elseif r >= 0.115
        rho = 1.0
        v1 = 0.0
        v2 = 0.0
    else
        rho = 1.0 + 9.0 * f
        v1 = -20.0 * f * dy
        v2 = 20.0 * f * dx
    end
    v3 = 0.0
    p = 1.0
    B1 = 5.0 / sqrt(4.0 * pi)
    B2 = 0.0
    B3 = 0.0
    psi = 0.0
    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end
initial_condition = initial_condition_rotor

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell)
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
solver = DGSEM(basis, surface_flux, volume_integral)

# Affine type mapping to take the [-1,1]^2 domain from the mesh file
# and put it onto the rotor domain [0,1]^2 and then warp it with a mapping
# as described in https://arxiv.org/abs/2012.12040
function mapping_twist(xi, eta)
    y = 0.5 * (eta + 1.0) +
        0.05 * cos(1.5 * pi * (2.0 * xi - 1.0)) * cos(0.5 * pi * (2.0 * eta - 1.0))
    x = 0.5 * (xi + 1.0) + 0.05 * cos(0.5 * pi * (2.0 * xi - 1.0)) * cos(2.0 * pi * y)
    return SVector(x, y)
end

mesh_file = Trixi.download("https://gist.githubusercontent.com/efaulhaber/63ff2ea224409e55ee8423b3a33e316a/raw/7db58af7446d1479753ae718930741c47a3b79b7/square_unstructured_2.inp",
                           joinpath(@__DIR__, "square_unstructured_2.inp"))

mesh = T8codeMesh(mesh_file, 2; polydeg = 4,
                  mapping = mapping_twist,
                  initial_refinement_level = 1)

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:all => boundary_condition)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.15)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

amr_indicator = IndicatorLöhner(semi,
                                variable = density_pressure)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 1,
                                      med_level = 3, med_threshold = 0.05,
                                      max_level = 5, max_threshold = 0.1)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

cfl = 0.5
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

# Finalize `T8codeMesh` to make sure MPI related objects in t8code are
# released before `MPI` finalizes.
!isinteractive() && finalize(mesh)
