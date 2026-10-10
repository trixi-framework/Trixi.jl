# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    CompressibleRANSDiffusion2D(equations; mu, Prandtl, Prandtl_turbulent = 0.9,
                                model = SpalartAllmarasNeg())

Parabolic part of the compressible Reynolds-averaged (Favre-averaged) Navier-Stokes equations
with a one-equation eddy-viscosity turbulence `model`.
The hyperbolic part `equations` is a [`PassiveTracerEquations`](@ref) wrapping
[`CompressibleEulerEquations2D`](@ref) with a single tracer, so that the conservative variables are
``(\rho, \rho v_1, \rho v_2, \rho e_{\text{total}}, \rho\tilde\nu)``.

With the eddy viscosity ``\mu_t`` of the `model`, the viscous stress tensor and heat flux of
[`CompressibleNavierStokesDiffusion2D`](@ref) use ``\mu + \mu_t`` and
``\kappa = \frac{\gamma}{\gamma - 1}\left(\frac{\mu}{\textrm{Pr}} + \frac{\mu_t}{\textrm{Pr}_t}\right)``
(with gas constant ``R = 1``). The parabolic flux of the turbulence variable is
```math
\frac{1}{\sigma} (\mu + \rho \tilde\nu f_n) \nabla\tilde\nu.
```
The remaining terms of the turbulence model (production, destruction, and the
non-conservative diffusion terms) depend on gradients and the wall distance and are provided by
[`SourceTermsSpalartAllmaras`](@ref), which has to be passed as `source_terms_parabolic`
to [`SemidiscretizationHyperbolicParabolic`](@ref).

The dynamic viscosity `mu` may be a constant or a function `mu(u, equations)` of the conservative
variables `u` (passed together with the hyperbolic equations).

Gradients are computed for the primitive variables ``(\rho, v_1, v_2, T, \tilde\nu)``
(`GradientVariablesPrimitive`).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct CompressibleRANSDiffusion2D{GradientVariables, RealT <: Real, Mu, Model,
                                   E <: AbstractEquations{2}} <:
       AbstractCompressibleRANSDiffusion{2, 5, GradientVariables}
    mu::Mu                    # molecular viscosity
    Pr::RealT                 # Prandtl number
    Pr_t::RealT               # turbulent Prandtl number
    kappa_over_mu::RealT      # gamma / ((gamma - 1) Pr)
    kappa_t_over_mu_t::RealT  # gamma / ((gamma - 1) Pr_t)
    max_visc_cond::RealT      # max(4/3, gamma / Pr, gamma / Pr_t) for `max_diffusivity`
    model::Model              # turbulence model, e.g., `SpalartAllmarasNeg`

    equations_hyperbolic::E   # `PassiveTracerEquations` of `CompressibleEulerEquations2D`
    gradient_variables::GradientVariables
end

function CompressibleRANSDiffusion2D(equations::PassiveTracerEquations{2, 5, 1,
                                                                       <:CompressibleEulerEquations2D};
                                     mu, Prandtl, Prandtl_turbulent = 0.9,
                                     model = SpalartAllmarasNeg())
    gradient_variables = GradientVariablesPrimitive()
    @unpack gamma, inv_gamma_minus_one = equations.flow_equations

    RealT = promote_type(typeof(gamma), typeof(Prandtl))
    Pr = convert(RealT, Prandtl)
    Pr_t = convert(RealT, Prandtl_turbulent)
    kappa_over_mu = gamma * inv_gamma_minus_one / Pr
    kappa_t_over_mu_t = gamma * inv_gamma_minus_one / Pr_t
    max_visc_cond = max(4 / 3, gamma / Pr, gamma / Pr_t, 1 / model.sigma)

    return CompressibleRANSDiffusion2D{typeof(gradient_variables), RealT, typeof(mu),
                                       typeof(model), typeof(equations)}(mu, Pr, Pr_t,
                                                                         kappa_over_mu,
                                                                         kappa_t_over_mu_t,
                                                                         max_visc_cond,
                                                                         model,
                                                                         equations,
                                                                         gradient_variables)
end

function Base.similar(equations::CompressibleRANSDiffusion2D,
                      ::Type{NewRealT}) where {NewRealT}
    mu = equations.mu isa Real ? convert(NewRealT, equations.mu) : equations.mu
    return CompressibleRANSDiffusion2D(similar(equations.equations_hyperbolic,
                                               NewRealT);
                                       mu = mu,
                                       Prandtl = convert(NewRealT, equations.Pr),
                                       Prandtl_turbulent = convert(NewRealT,
                                                                   equations.Pr_t),
                                       model = similar(equations.model, NewRealT))
end

"""
    cons2prim_temperature(u, equations::CompressibleRANSDiffusion2D)

Convert conservative variables `u` to `(rho, v1, v2, T, nu_tilde)` with temperature `T = p / rho`.
"""
@inline function cons2prim_temperature(u, equations::CompressibleRANSDiffusion2D)
    rho, rho_v1, rho_v2, rho_e_total, rho_nu_tilde = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gamma - 1) * (rho_e_total - 0.5f0 * (rho_v1 * v1 + rho_v2 * v2))
    return SVector(rho, v1, v2, p / rho, rho_nu_tilde / rho)
end

# Conservative variables from `(rho, v1, v2, T, nu_tilde)`
@inline function prim_temperature2cons(prim, equations::CompressibleRANSDiffusion2D)
    rho, v1, v2, T, nu_tilde = prim
    rho_e_total = rho * T * equations.inv_gamma_minus_one +
                  0.5f0 * rho * (v1^2 + v2^2)
    return SVector(rho, rho * v1, rho * v2, rho_e_total, rho * nu_tilde)
end

function flux(u, gradients, orientation::Integer,
              equations::CompressibleRANSDiffusion2D)
    # `u` are the transformed variables specified by `gradient_variable_transformation`
    prim = convert_transformed_to_primitive(u, equations)
    rho, v1, v2, _, nu_tilde = prim
    _, dv1dx, dv2dx, dTdx, dnudx = convert_derivative_to_primitive(prim, gradients[1],
                                                                   equations)
    _, dv1dy, dv2dy, dTdy, dnudy = convert_derivative_to_primitive(prim, gradients[2],
                                                                   equations)

    mu = dynamic_viscosity_prim(prim, equations)
    mu_t = sa_eddy_viscosity(rho, nu_tilde, mu, equations.model)
    mu_eff = mu + mu_t
    kappa = equations.kappa_over_mu * mu + equations.kappa_t_over_mu_t * mu_t
    diffusivity_nu = sa_diffusivity(rho, nu_tilde, mu, equations.model)

    # Components of the viscous stress tensor (without the viscosity)
    tau_11 = (4 * dv1dx - 2 * dv2dy) / 3
    tau_12 = dv1dy + dv2dx
    tau_22 = (4 * dv2dy - 2 * dv1dx) / 3

    if orientation == 1
        f2 = tau_11 * mu_eff
        f3 = tau_12 * mu_eff
        f4 = v1 * f2 + v2 * f3 + kappa * dTdx
        f5 = diffusivity_nu * dnudx
        return SVector(zero(f2), f2, f3, f4, f5)
    else # if orientation == 2
        g2 = tau_12 * mu_eff
        g3 = tau_22 * mu_eff
        g4 = v1 * g2 + v2 * g3 + kappa * dTdy
        g5 = diffusivity_nu * dnudy
        return SVector(zero(g2), g2, g3, g4, g5)
    end
end

# Magnitude of the vorticity from the gradients of `(rho, v1, v2, T, nu_tilde)`
@inline function vorticity_magnitude(gradients_prim,
                                     ::CompressibleRANSDiffusion2D)
    _, _, dv2dx, _, _ = gradients_prim[1]
    _, dv1dy, _, _, _ = gradients_prim[2]
    return abs(dv2dx - dv1dy)
end
end # @muladd
