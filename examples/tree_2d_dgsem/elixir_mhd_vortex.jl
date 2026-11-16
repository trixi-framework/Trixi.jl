using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

equations = IdealGlmMhdEquations2D(5 / 3)

# Vortex parameters in Heaviside-Lorentz units, see the docstring below.

# Strength of the velocity perturbation
vortex_kappa() = 1.0
# Strength of the magnetic perturbation
vortex_mu() = 1.0
advection_velocity() = (1.0, 1.0)

# Background pressure - must be large enough to ensure that the pressure stays positive.
p_0() = 1.0

# Balsara (2004) uses a domain of size 10x10 - for the high-order DGSEM scheme,
# this domain is too small to conduct e.g. a convergence study,
# as the Gaussian is not small enough in comparison to the discretization error
domain_length() = 20.0

@doc raw"""
    initial_condition_vortex_balsara(x, t, equations::IdealGlmMhdEquations2D)

Smooth magnetized vortex, which is an exact stationary solution of the ideal MHD
equations in its rest frame. The setup is taken from
- Dinshaw S. Balsara (2004)
  Second-Order-accurate Schemes for Magnetohydrodynamics with Divergence-free Reconstruction
  [DOI: 10.1086/381377](https://doi.org/10.1086/381377)

**Units:**
The ideal MHD equations in Trixi.jl are written in Heaviside-Lorentz-type units, i.e.,
the magnetic pressure is ``|\boldsymbol{B}|^2 / 2`` (vacuum permeability ``\mu_0 = 1``).
Balsara (2004) uses Gaussian units with magnetic pressure ``|\boldsymbol{B}|^2 / (8 \pi)``,
which results in the additional factor ``1 / (8 \pi)``
in front of the magnetic part of the pressure perturbation in the paper.
Both formulations describe the same flow if the magnetic field is rescaled as
``\boldsymbol{B}_\mathrm{HL} = \boldsymbol{B}_\mathrm{Gauss} / \sqrt{4 \pi}``, i.e.,
``\mu = \mu_\mathrm{Balsara} / \sqrt{4 \pi}``.
Balsara's choice ``\kappa = 1`` and ``\mu_\mathrm{Balsara} = \sqrt{4 \pi}``
thus corresponds to the default values ``\kappa = 1`` and ``\mu = 1`` here.
"""
function initial_condition_vortex_balsara(x, t, equations::IdealGlmMhdEquations2D)
    RealT = eltype(x)
    u0 = convert.(RealT, advection_velocity())
    l_box = convert(RealT, domain_length())

    # Translate back to the t = 0 position and wrap into the periodic box
    x1 = mod(x[1] - u0[1] * t + l_box / 2, l_box) - l_box / 2
    x2 = mod(x[2] - u0[2] * t + l_box / 2, l_box) - l_box / 2

    r2 = x1^2 + x2^2
    g = exp(0.5f0 * (1 - r2))
    a = convert(RealT, vortex_kappa()) / (2 * convert(RealT, pi)) # cf. eq. (6.1)
    b = convert(RealT, vortex_mu()) / (2 * convert(RealT, pi)) # cf. eq. (6.2)

    rho = one(RealT)
    v1 = u0[1] - a * g * x2 # cf. eq. (6.1)
    v2 = u0[2] + a * g * x1 # cf. eq. (6.1)
    v3 = zero(RealT)
    B1 = -b * g * x2 # cf. eq. (6.2)
    B2 = b * g * x1  # cf. eq. (6.2)
    B3 = zero(RealT)
    # cf. eq. (6.5)
    p = (convert(RealT, p_0()) -
         0.5f0 * exp(1 - r2) * (a^2 + b^2 * (r2 - 1)))
    psi = zero(RealT)

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end
initial_condition = initial_condition_vortex_balsara

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell)
volume_flux = (flux_derigs_etal, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3,
               surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-domain_length() / 2, -domain_length() / 2)
coordinates_max = (domain_length() / 2, domain_length() / 2)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition_periodic)

tspan = (0.0, 20.0) # one pass of the vortex through the periodic domain
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

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
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
