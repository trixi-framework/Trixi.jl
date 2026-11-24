using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

equations = IdealGlmMhdEquations2D(1.4)

"""
    initial_condition_blast_wave(x, t, equations::IdealGlmMhdEquations2D)

An MHD blast wave taken from
- Dominik Derigs, Gregor J. Gassner, Stefanie Walch & Andrew R. Winters (2018)
  Entropy Stable Finite Volume Approximations for Ideal Magnetohydrodynamics
  [arXiv: 1708.03537](https://arxiv.org/abs/1708.03537)
  [DOI: 10.1365/s13291-018-0178-9](https://doi.org/10.1365/s13291-018-0178-9)
"""
function initial_condition_blast_wave(x, t, equations::IdealGlmMhdEquations2D)
    # setup taken from Derigs et al. DMV article (2018)
    # domain must be [-0.5, 0.5] x [-0.5, 0.5], γ = 1.4
    RealT = eltype(x)
    r = sqrt(x[1]^2 + x[2]^2)
    f = (convert(RealT, 0.1) - r) / convert(RealT, 0.01)
    if r <= RealT(0.09)
        p = convert(RealT, 1000)
    elseif r >= RealT(0.1)
        p = convert(RealT, 0.1)
    else
        p = convert(RealT, 0.1) + convert(RealT, 999.9) * f
    end
    rho = 1
    v1 = 0
    v2 = 0
    v3 = 0
    B1 = 100 / sqrt(4 * convert(RealT, pi))
    B2 = 0
    B3 = 0
    psi = 0
    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end
initial_condition = initial_condition_blast_wave

# Entropy-conservative flux differencing without shock capturing. Positivity of density
# and pressure is enforced by the positivity-preserving limiters below.
surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-0.5, -0.5)
coordinates_max = (0.5, 0.5)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.01)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# A large CFL number is used deliberately so that cell averages leave the admissible set
# and the global Liu-Zhang limiter is triggered.
cfl = 1.2
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

# The local Zhang-Shu limiter enforces positivity of density and pressure at the nodes
# provided that the cell averages are admissible, which is enforced by the global
# Liu-Zhang limiter. For the ideal GLM-MHD equations, the admissible set is convex and the
# projection onto it uses the slicing algorithm of Liu, Shu, Zhang (2026).
local_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1e-8, 1e-8),
                                                     variables = (Trixi.density, pressure))
global_limiter! = PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
                                                      record_davis_yin_iterations = true)

ode_solver = CarpenterKennedy2N54(; stage_limiter! = global_limiter!,
                                  step_limiter! = global_limiter!,
                                  williamson_condition = false, thread = Trixi.Threaded())

sol = solve(ode, ode_solver;
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
