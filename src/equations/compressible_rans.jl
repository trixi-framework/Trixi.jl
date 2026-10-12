# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Dimension-independent parts of the compressible RANS equations with the
# Spalart-Allmaras model: the turbulence model, the model source terms, and the boundary
# conditions. The dimension-specific parts (variables, viscous fluxes, vorticity) are in
# `compressible_rans_2d.jl`.

@doc raw"""
    SpalartAllmarasNeg(RealT = Float64; ct3 = 1.2)

Negative Spalart-Allmaras (SA-neg) one-equation turbulence model for use with
[`CompressibleRANSDiffusion2D`](@ref) and [`SourceTermsSpalartAllmaras`](@ref).
The model follows
- S. R. Allmaras, F. T. Johnson, P. R. Spalart (2012)
  Modifications and clarifications for the implementation of the Spalart-Allmaras turbulence model
  ICCFD7-1902
  [https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf](https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf)

For ``\tilde\nu \geq 0`` this is the standard model (version Ia, including the ``f_{t2}`` term
and the modified vorticity ``\tilde S`` of eq. (12) of the reference). For ``\tilde\nu < 0`` the
negative continuation (eqs. (14), (21), (22)) is used, with zero eddy viscosity.
The trip term ``f_{t1}`` is not included. Setting `ct3 = 0` gives the SA-noft2 variant.

The conservative compressible form (eq. (9) of the reference) is used, including the term
``-\frac{1}{\sigma}(\nu + \tilde\nu f_n) \nabla\rho \cdot \nabla\tilde\nu``.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct SpalartAllmarasNeg{RealT <: Real}
    cb1::RealT
    cb2::RealT
    sigma::RealT
    kappa::RealT
    cw1::RealT
    cw2::RealT
    cw3::RealT
    cv1::RealT
    cv2::RealT
    cv3::RealT
    ct3::RealT
    ct4::RealT
    cn1::RealT
    rlim::RealT
end

function SpalartAllmarasNeg(RealT = Float64; ct3 = 1.2)
    cb1 = convert(RealT, 0.1355)
    cb2 = convert(RealT, 0.622)
    sigma = convert(RealT, 2 / 3)
    kappa = convert(RealT, 0.41)
    cw1 = cb1 / kappa^2 + (1 + cb2) / sigma
    return SpalartAllmarasNeg{RealT}(cb1, cb2, sigma, kappa, cw1,
                                     convert(RealT, 0.3), # cw2
                                     convert(RealT, 2), # cw3
                                     convert(RealT, 7.1), # cv1
                                     convert(RealT, 0.7), # cv2
                                     convert(RealT, 0.9), # cv3
                                     convert(RealT, ct3),
                                     convert(RealT, 0.5), # ct4
                                     convert(RealT, 16), # cn1
                                     convert(RealT, 10)) # rlim
end

function Base.similar(model::SpalartAllmarasNeg, ::Type{NewRealT}) where {NewRealT}
    return SpalartAllmarasNeg{NewRealT}((convert(NewRealT, getfield(model, f))
                                         for f in fieldnames(SpalartAllmarasNeg))...)
end

# Viscous damping function f_v1 of the eddy viscosity
@inline function sa_fv1(chi, model::SpalartAllmarasNeg)
    chi3 = chi^3
    return chi3 / (chi3 + model.cv1^3)
end

# Modification of the diffusion coefficient for negative nu_tilde, eq. (21) of ICCFD7-1902
@inline function sa_fn(chi, model::SpalartAllmarasNeg)
    if chi >= 0
        return one(chi)
    else
        chi3 = chi^3
        return (model.cn1 + chi3) / (model.cn1 - chi3)
    end
end

# Turbulent (eddy) viscosity μ_t = ρ ν̃ f_v1 for ν̃ ≥ 0 and zero otherwise
@inline function sa_eddy_viscosity(rho, nu_tilde, mu, model::SpalartAllmarasNeg)
    if nu_tilde > 0
        chi = rho * nu_tilde / mu
        return rho * nu_tilde * sa_fv1(chi, model)
    else
        return zero(nu_tilde)
    end
end

# Diffusion coefficient (μ + ρ ν̃ f_n) / σ of the ν̃ equation
@inline function sa_diffusivity(rho, nu_tilde, mu, model::SpalartAllmarasNeg)
    chi = rho * nu_tilde / mu
    return (mu + rho * nu_tilde * sa_fn(chi, model)) / model.sigma
end

# Production minus destruction (P - D) of the kinematic ν̃ equation, i.e., without the factor ρ.
# `vorticity` is the magnitude of the vorticity and `d` the distance to the nearest wall.
@inline function sa_production_destruction(nu_tilde, nu, vorticity, d,
                                           model::SpalartAllmarasNeg)
    @unpack cb1, kappa, cw1, cw2, cw3, cv2, cv3, ct3, ct4, rlim = model

    # The production and destruction terms are singular at the wall itself.
    # There, the Dirichlet condition ν̃ = 0 holds and both terms vanish.
    if d <= 0
        return zero(nu_tilde)
    end

    inv_d2 = 1 / d^2
    if nu_tilde >= 0
        chi = nu_tilde / nu
        fv1 = sa_fv1(chi, model)
        fv2 = 1 - chi / (1 + chi * fv1)
        ft2 = ct3 * exp(-ct4 * chi^2)

        # Modified vorticity, eqs. (11) and (12) of ICCFD7-1902
        inv_kappa2_d2 = inv_d2 / kappa^2
        S_bar = nu_tilde * fv2 * inv_kappa2_d2
        if S_bar >= -cv2 * vorticity
            S_tilde = vorticity + S_bar
        else
            S_tilde = vorticity +
                      vorticity * (cv2^2 * vorticity + cv3 * S_bar) /
                      ((cv3 - 2 * cv2) * vorticity - S_bar)
        end

        if S_tilde > 0
            r = min(nu_tilde * inv_kappa2_d2 / S_tilde, rlim)
        else
            r = rlim
        end
        g = r + cw2 * (r^6 - r)
        cw3_6 = cw3^6
        fw = g * ((1 + cw3_6) / (g^6 + cw3_6))^(one(g) / 6)

        production = cb1 * (1 - ft2) * S_tilde * nu_tilde
        destruction = (cw1 * fw - cb1 / kappa^2 * ft2) * nu_tilde^2 * inv_d2
    else
        # Negative SA model, eq. (22) of ICCFD7-1902
        production = cb1 * (1 - ct3) * vorticity * nu_tilde
        destruction = -cw1 * nu_tilde^2 * inv_d2
    end

    return production - destruction
end

# Common supertype of the compressible RANS equations in different dimensions, currently
# `CompressibleRANSDiffusion2D`.
# The conservative variables are `(rho, rho v_1, ..., rho v_NDIMS, rho e_total, rho nu_tilde)`
# and the primitive variables used for the parabolic terms are
# `(rho, v_1, ..., v_NDIMS, T, nu_tilde)`. The concrete types implement
# `cons2prim_temperature`, `prim_temperature2cons`, `flux`, and `vorticity_magnitude`.
abstract type AbstractCompressibleRANSDiffusion{NDIMS, NVARS, GradientVariables} <:
              AbstractEquationsParabolic{NDIMS, NVARS, GradientVariables} end

const CompressibleRANSDiffusionPrimitive = AbstractCompressibleRANSDiffusion{<:Any,
                                                                             <:Any,
                                                                             GradientVariablesPrimitive}

@inline function Base.getproperty(equations::AbstractCompressibleRANSDiffusion,
                                  field::Symbol)
    if field === :gamma || field === :inv_gamma_minus_one
        return getproperty(getfield(equations, :equations_hyperbolic).flow_equations,
                           field)
    else
        return getfield(equations, field)
    end
end

function varnames(variable_mapping, equations::AbstractCompressibleRANSDiffusion)
    return varnames(variable_mapping, equations.equations_hyperbolic)
end

function gradient_variable_transformation(::CompressibleRANSDiffusionPrimitive)
    return cons2prim_temperature
end

@inline have_constant_diffusivity(::AbstractCompressibleRANSDiffusion) = False()

@inline function cons2prim(u, equations::AbstractCompressibleRANSDiffusion)
    return cons2prim(u, equations.equations_hyperbolic)
end

@inline function prim2cons(u, equations::AbstractCompressibleRANSDiffusion)
    return prim2cons(u, equations.equations_hyperbolic)
end

@inline function temperature(u, equations::AbstractCompressibleRANSDiffusion)
    return cons2prim_temperature(u, equations)[ndims(equations) + 2]
end

@inline function velocity(u, equations::AbstractCompressibleRANSDiffusion)
    rho = u[1]
    return SVector(ntuple(@inline(i->u[i + 1] / rho), Val(ndims(equations))))
end

@inline function convert_transformed_to_primitive(u_transformed,
                                                  equations::CompressibleRANSDiffusionPrimitive)
    return u_transformed
end

# Convert the gradient of the transformed variables to the gradient of
# `(rho, v_1, ..., v_NDIMS, T, nu_tilde)`.
# The first argument are the primitive variables `(rho, v_1, ..., v_NDIMS, T, nu_tilde)`.
@inline function convert_derivative_to_primitive(prim, gradient,
                                                 ::CompressibleRANSDiffusionPrimitive)
    return gradient
end

@inline function dynamic_viscosity_prim(prim,
                                        equations::AbstractCompressibleRANSDiffusion)
    return dynamic_viscosity(prim2cons_mu(prim, equations.mu, equations), equations.mu,
                             equations.equations_hyperbolic)
end
# Avoid the conversion to conservative variables for constant viscosity
@inline prim2cons_mu(prim, ::Real, equations) = prim
@inline prim2cons_mu(prim, mu, equations) = prim_temperature2cons(prim, equations)

@inline function max_diffusivity(u, equations::AbstractCompressibleRANSDiffusion)
    rho = u[1]
    nu_tilde = u[nvariables(equations)] / rho
    mu = dynamic_viscosity(u, equations.mu, equations.equations_hyperbolic)
    mu_t = sa_eddy_viscosity(rho, nu_tilde, mu, equations.model)
    chi = rho * nu_tilde / mu
    mu_nu = mu + rho * nu_tilde * sa_fn(chi, equations.model)
    return max(mu + mu_t, mu_nu) / rho * equations.max_visc_cond
end

@doc raw"""
    eddy_viscosity(u, equations::CompressibleRANSDiffusion2D)

Eddy viscosity ``\mu_t`` of the turbulence model for the conservative variables `u`.
"""
@inline function eddy_viscosity(u, equations::AbstractCompressibleRANSDiffusion)
    rho = u[1]
    mu = dynamic_viscosity(u, equations.mu, equations.equations_hyperbolic)
    return sa_eddy_viscosity(rho, u[nvariables(equations)] / rho, mu, equations.model)
end

@doc raw"""
    SourceTermsSpalartAllmaras(wall_distance)

Source terms of the Spalart-Allmaras turbulence model for [`CompressibleRANSDiffusion2D`](@ref),
to be passed as `source_terms_parabolic` to [`SemidiscretizationHyperbolicParabolic`](@ref).
They contain production and destruction and the non-conservative parts of the diffusion,
```math
\rho (P - D) + \frac{c_{b2}}{\sigma} \rho \nabla\tilde\nu \cdot \nabla\tilde\nu
- \frac{1}{\sigma} (\nu + \tilde\nu f_n) \nabla\rho \cdot \nabla\tilde\nu,
```
see [`SpalartAllmarasNeg`](@ref). The production uses the magnitude of the vorticity.

The distance to the nearest (no-slip) wall is given by the function `wall_distance(x)` of the
coordinates `x`.
Where the wall distance is zero, production and destruction are set to zero.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct SourceTermsSpalartAllmaras{WallDistance}
    wall_distance::WallDistance
end

@inline function (source::SourceTermsSpalartAllmaras)(u, gradients, x, t,
                                                      equations::AbstractCompressibleRANSDiffusion)
    return spalart_allmaras_source(u, gradients, source.wall_distance(x), equations)
end

# Source terms of the SA model for the wall distance `d`
@inline function spalart_allmaras_source(u, gradients, d,
                                         equations::AbstractCompressibleRANSDiffusion)
    @unpack model = equations
    prim = cons2prim_temperature(u, equations)
    rho = prim[1]
    nu_tilde = prim[end]

    # Gradients of `(rho, v_1, ..., v_NDIMS, T, nu_tilde)`, one per coordinate direction
    gradients_prim = map(gradient -> convert_derivative_to_primitive(prim, gradient,
                                                                     equations),
                         gradients)
    grad_rho = SVector(map(first, gradients_prim))
    grad_nu_tilde = SVector(map(last, gradients_prim))

    mu = dynamic_viscosity(u, equations.mu, equations.equations_hyperbolic)
    nu = mu / rho
    vorticity = vorticity_magnitude(gradients_prim, equations)

    source_nu = rho * sa_production_destruction(nu_tilde, nu, vorticity, d, model)
    source_nu += model.cb2 / model.sigma * rho * dot(grad_nu_tilde, grad_nu_tilde)
    source_nu += sa_density_gradient_source(nu_tilde, nu, grad_rho, grad_nu_tilde,
                                            model)

    return source_last_variable(source_nu, equations)
end

# Non-conservative diffusion term -(ν + ν̃ f_n) / σ ∇ρ ⋅ ∇ν̃ of the conservative compressible
# form of the ν̃ equation (eq. (9) of ICCFD7-1902)
@inline function sa_density_gradient_source(nu_tilde, nu, grad_rho, grad_nu_tilde,
                                            model::SpalartAllmarasNeg)
    chi = nu_tilde / nu
    return -(nu + nu_tilde * sa_fn(chi, model)) / model.sigma *
           dot(grad_rho, grad_nu_tilde)
end

# Vector of length `nvariables(equations)` with `source` as last entry and zeros otherwise
@inline function source_last_variable(source, equations)
    return SVector(ntuple(@inline(v->v == nvariables(equations) ? source : zero(source)),
                          Val(nvariables(equations))))
end

###############################################################################
# Boundary conditions.
# The gradient boundary conditions return the boundary values of the transformed variables,
# the divergence boundary conditions return the normal parabolic flux.
# For the turbulence variable, no-slip walls use ν̃ = 0 (Dirichlet) and slip walls / symmetry
# planes use ∂ν̃/∂n = 0 (Neumann).

@inline function transform_primitive_bc(prim,
                                        equations::CompressibleRANSDiffusionPrimitive)
    return prim
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      u_inner,
                                                                                      normal::AbstractVector,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Gradient,
                                                                                      equations::AbstractCompressibleRANSDiffusion)
    v = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t,
                                                                               equations)
    prim = convert_transformed_to_primitive(u_inner, equations)
    rho = prim[1]
    T = prim[ndims(equations) + 2]
    return transform_primitive_bc(SVector(rho, v..., T, zero(T)), equations)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Isothermal})(flux_inner,
                                                                                       u_inner,
                                                                                       normal::AbstractVector,
                                                                                       x,
                                                                                       t,
                                                                                       operator_type::Gradient,
                                                                                       equations::AbstractCompressibleRANSDiffusion)
    v = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t,
                                                                               equations)
    T = boundary_condition.boundary_condition_heat_flux.boundary_value_function(x, t,
                                                                                equations)
    rho = convert_transformed_to_primitive(u_inner, equations)[1]
    return transform_primitive_bc(SVector(rho, v..., T, zero(T)), equations)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      u_inner,
                                                                                      normal::AbstractVector,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Divergence,
                                                                                      equations::AbstractCompressibleRANSDiffusion)
    NDIMS = ndims(equations)
    normal_heat_flux = boundary_condition.boundary_condition_heat_flux.boundary_value_normal_flux_function(x,
                                                                                                           t,
                                                                                                           equations)
    v = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t,
                                                                               equations)
    # Normal viscous stresses (fluxes of the momentum equations)
    tau_n = SVector(ntuple(@inline(i->flux_inner[i + 1]), Val(NDIMS)))
    normal_energy_flux = dot(SVector(v), tau_n) + normal_heat_flux
    return SVector(ntuple(@inline(i->i == NDIMS + 2 ? normal_energy_flux :
                                     flux_inner[i]),
                          Val(nvariables(equations))))
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Isothermal})(flux_inner,
                                                                                       u_inner,
                                                                                       normal::AbstractVector,
                                                                                       x,
                                                                                       t,
                                                                                       operator_type::Divergence,
                                                                                       equations::AbstractCompressibleRANSDiffusion)
    return flux_inner
end

# Slip wall. Should be used with `boundary_condition_slip_wall` for the hyperbolic part.
# As for `CompressibleNavierStokesDiffusion2D`, the whole viscous traction (including the
# normal stress) is set to zero at the wall. Thus, this is not a true symmetry plane for
# viscous flows, which would only remove the tangential traction.
@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:Slip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      u_inner,
                                                                                      normal::AbstractVector,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Gradient,
                                                                                      equations::AbstractCompressibleRANSDiffusion)
    NDIMS = ndims(equations)
    prim = convert_transformed_to_primitive(u_inner, equations)
    v = ntuple(@inline(i->prim[i + 1]), Val(NDIMS))
    v_outer = velocity_symmetry_plane(normal, v...)
    return transform_primitive_bc(SVector(prim[1], v_outer..., prim[NDIMS + 2],
                                          prim[NDIMS + 3]),
                                  equations)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:Slip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      u_inner,
                                                                                      normal::AbstractVector,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Divergence,
                                                                                      equations::AbstractCompressibleRANSDiffusion)
    NDIMS = ndims(equations)
    normal_heat_flux = boundary_condition.boundary_condition_heat_flux.boundary_value_normal_flux_function(x,
                                                                                                           t,
                                                                                                           equations)
    z = zero(normal_heat_flux)
    return SVector(ntuple(@inline(i->i == 1 ? flux_inner[1] :
                                     (i == NDIMS + 2 ? normal_heat_flux : z)),
                          Val(nvariables(equations))))
end

@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner,
                                                                  u_inner,
                                                                  normal::AbstractVector,
                                                                  x, t,
                                                                  operator_type::Gradient,
                                                                  equations::AbstractCompressibleRANSDiffusion)
    u_boundary = boundary_condition.boundary_value_function(x, t, equations)
    return gradient_variable_transformation(equations)(u_boundary, equations)
end

@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner,
                                                                  u_inner,
                                                                  normal::AbstractVector,
                                                                  x, t,
                                                                  operator_type::Divergence,
                                                                  equations::AbstractCompressibleRANSDiffusion)
    return flux_inner
end
end # @muladd
