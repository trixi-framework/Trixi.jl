# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    CompressibleLES3D(equations; mu, Prandtl, R = 1, subgrid_model = Smagorinsky())

Parabolic terms of the filtered compressible Navier-Stokes equations for large eddy
simulations (LES) together with the advective terms of the
[`CompressibleEulerEquations3D`](@ref) `equations`, i.e.,
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
\rho \\ \rho \mathbf{v} \\ \rho e_{\text{total}}
\end{pmatrix}
+
\nabla \cdot
\begin{pmatrix}
 \rho \mathbf{v} \\ \rho \mathbf{v}\mathbf{v}^T + p \underline{I} \\ (\rho e_{\text{total}} + p) \mathbf{v}
\end{pmatrix}
=
\nabla \cdot
\begin{pmatrix}
0 \\ \underline{\tau} \\ \underline{\tau}\mathbf{v} + (\kappa + \kappa_t) \nabla T
\end{pmatrix}
```
with the viscous stress tensor
```math
\underline{\tau} = (\mu + \mu_t) \left(\nabla\mathbf{v} + \left(\nabla\mathbf{v}\right)^T
                   - \frac{2}{3} \left(\nabla\cdot\mathbf{v}\right)\underline{I}\right)
```
and the temperature ``T = p / (R \rho)``. The eddy viscosity ``\mu_t`` is given by the
subgrid-scale model `subgrid_model`, e.g., [`Smagorinsky`](@ref), and the eddy conductivity
is ``\kappa_t = c_p \mu_t / \mathrm{Pr}_t`` with the turbulent Prandtl number
``\mathrm{Pr}_t`` of the `subgrid_model`. As for the
[`CompressibleNavierStokesDiffusion3D`](@ref), the thermal conductivity is
``\kappa = c_p \mu / \mathrm{Pr}`` with the Prandtl number `Prandtl` and the specific
heat capacity ``c_p = \gamma R / (\gamma - 1)``, and the dynamic viscosity `mu` may be a
constant or a function `mu(u, equations)` of the primitive variables
`u = (rho, v1, v2, v3, T)`, whose gradients are used.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct CompressibleLES3D{RealT <: Real, Mu, E <: AbstractCompressibleEulerEquations{3},
                         SubgridModel <: AbstractSubgridModel} <:
       AbstractLESDiffusion{3, 5, GradientVariablesPrimitive}
    R::RealT                # specific gas constant
    mu::Mu                  # dynamic viscosity
    Pr::RealT               # Prandtl number
    kappa_over_mu::RealT    # c_p / Pr
    max_visc_cond::RealT    # max(4/3, gamma/Pr) used for `max_diffusivity`
    equations_hyperbolic::E # `CompressibleEulerEquations3D`
    subgrid_model::SubgridModel
end

function CompressibleLES3D(equations::CompressibleEulerEquations3D; mu, Prandtl, R = 1,
                           subgrid_model = Smagorinsky())
    (; gamma, inv_gamma_minus_one) = equations
    Pr = promote_type(typeof(gamma), typeof(Prandtl), typeof(R))(Prandtl)
    R = convert(typeof(Pr), R)
    kappa_over_mu = gamma * inv_gamma_minus_one * R / Pr
    max_visc_cond = max(convert(typeof(Pr), 4) / 3, gamma / Pr)

    return CompressibleLES3D(R, mu, Pr, kappa_over_mu, max_visc_cond, equations,
                             subgrid_model)
end

function Base.similar(equations::CompressibleLES3D, ::Type{NewRealT}) where {NewRealT}
    mu = equations.mu isa Real ? convert(NewRealT, equations.mu) : equations.mu
    return CompressibleLES3D(similar(equations.equations_hyperbolic, NewRealT); mu,
                             Prandtl = convert(NewRealT, equations.Pr),
                             R = convert(NewRealT, equations.R),
                             subgrid_model = equations.subgrid_model)
end

# Forward the accesses of `gamma` and `inv_gamma_minus_one` to the hyperbolic equations
@inline function Base.getproperty(equations::CompressibleLES3D, field::Symbol)
    if field === :gamma || field === :inv_gamma_minus_one
        return getproperty(getfield(equations, :equations_hyperbolic), field)
    else
        return getfield(equations, field)
    end
end

@inline function Base.propertynames(equations::CompressibleLES3D, private::Bool = false)
    return (:gamma, :inv_gamma_minus_one, fieldnames(typeof(equations))...)
end

# The conversions of the hyperbolic equations are available, e.g., for boundary conditions
@inline cons2prim(u, equations::CompressibleLES3D) = cons2prim(u,
                                                               equations.equations_hyperbolic)
@inline prim2cons(u, equations::CompressibleLES3D) = prim2cons(u,
                                                               equations.equations_hyperbolic)

function varnames(variable_mapping, equations::CompressibleLES3D)
    return varnames(variable_mapping, equations.equations_hyperbolic)
end

gradient_variable_transformation(::CompressibleLES3D) = cons2prim_temperature

# Primitive variables `(rho, v1, v2, v3, T)` of the conservative variables `u`
@inline function cons2prim_temperature(u, equations::CompressibleLES3D)
    rho, rho_v1, rho_v2, rho_v3, _ = u
    return SVector(rho, rho_v1 / rho, rho_v2 / rho, rho_v3 / rho,
                   temperature(u, equations))
end

@inline function temperature(u, equations::CompressibleLES3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e_total = u
    p = (equations.gamma - 1) *
        (rho_e_total - 0.5f0 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho)
    return p / (rho * equations.R)
end

# Largest eigenvalue of the molecular diffusivity matrix, see
# `CompressibleNavierStokesDiffusion3D`
@inline function max_diffusivity(u, equations::CompressibleLES3D)
    return dynamic_viscosity_conservative(u, equations) / u[1] * equations.max_visc_cond
end

@inline function vorticity(u, gradients, equations::CompressibleLES3D)
    dv_dx, dv_dy, dv_dz = convert_derivative_to_velocity_gradients(u, gradients,
                                                                   equations)
    return SVector(dv_dy[3] - dv_dz[2], dv_dz[1] - dv_dx[3], dv_dx[2] - dv_dy[1])
end

@inline function enstrophy(u, gradients, equations::CompressibleLES3D)
    omega = vorticity(u, gradients, equations)
    return 0.5f0 * u[1] * (omega[1]^2 + omega[2]^2 + omega[3]^2)
end

# Interface of `AbstractLESDiffusion` for the primitive variables `(rho, v1, v2, v3, T)`
# and their gradients
@inline convert_transformed_to_density(u, equations::CompressibleLES3D) = u[1]

@inline function convert_transformed_to_velocity(u, equations::CompressibleLES3D)
    return SVector(u[2], u[3], u[4])
end

@inline function convert_derivative_to_velocity_gradients(u, gradients,
                                                          equations::CompressibleLES3D)
    return map(gradient -> SVector(gradient[2], gradient[3], gradient[4]), gradients)
end

# Viscosity `mu + mu_t` and thermal conductivity `kappa + kappa_t` with the eddy
# conductivity `kappa_t = c_p mu_t / Prandtl_turbulent`, where `kappa_over_mu * Pr = c_p`
@inline function diffusivities(u, mu_t, equations::CompressibleLES3D)
    (; Pr, kappa_over_mu) = equations
    mu = dynamic_viscosity(u, equations)
    kappa = kappa_over_mu * (mu + Pr / equations.subgrid_model.Prandtl_turbulent * mu_t)
    return SVector(mu + mu_t, kappa)
end

@inline function flux(u, gradients, orientation::Integer, diffusivities,
                      equations::CompressibleLES3D)
    mu, kappa = diffusivities
    _, v1, v2, v3, _ = u
    _, dv1dx, dv2dx, dv3dx, dTdx = gradients[1]
    _, dv1dy, dv2dy, dv3dy, dTdy = gradients[2]
    _, dv1dz, dv2dz, dv3dz, dTdz = gradients[3]

    # Components of the viscous stress tensor divided by the viscosity
    tau_11 = (4 * dv1dx - 2 * (dv2dy + dv3dz)) / 3
    tau_22 = (4 * dv2dy - 2 * (dv1dx + dv3dz)) / 3
    tau_33 = (4 * dv3dz - 2 * (dv1dx + dv2dy)) / 3
    tau_12 = dv1dy + dv2dx
    tau_13 = dv1dz + dv3dx
    tau_23 = dv2dz + dv3dy

    if orientation == 1
        return SVector(0, tau_11 * mu, tau_12 * mu, tau_13 * mu,
                       (v1 * tau_11 + v2 * tau_12 + v3 * tau_13) * mu + kappa * dTdx)
    elseif orientation == 2
        return SVector(0, tau_12 * mu, tau_22 * mu, tau_23 * mu,
                       (v1 * tau_12 + v2 * tau_22 + v3 * tau_23) * mu + kappa * dTdy)
    else # if orientation == 3
        return SVector(0, tau_13 * mu, tau_23 * mu, tau_33 * mu,
                       (v1 * tau_13 + v2 * tau_23 + v3 * tau_33) * mu + kappa * dTdz)
    end
end

# No-slip walls, see `CompressibleNavierStokesDiffusion3D`
@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      u_inner,
                                                                                      normal::AbstractVector,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Gradient,
                                                                                      equations::CompressibleLES3D)
    v1, v2, v3 = boundary_condition.boundary_condition_velocity.boundary_value_function(x,
                                                                                        t,
                                                                                        equations)
    return SVector(u_inner[1], v1, v2, v3, u_inner[5])
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      u_inner,
                                                                                      normal::AbstractVector,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Divergence,
                                                                                      equations::CompressibleLES3D)
    normal_heat_flux = boundary_condition.boundary_condition_heat_flux.boundary_value_normal_flux_function(x,
                                                                                                           t,
                                                                                                           equations)
    v1, v2, v3 = boundary_condition.boundary_condition_velocity.boundary_value_function(x,
                                                                                        t,
                                                                                        equations)
    _, tau_1n, tau_2n, tau_3n, _ = flux_inner
    normal_energy_flux = v1 * tau_1n + v2 * tau_2n + v3 * tau_3n + normal_heat_flux
    return SVector(flux_inner[1], flux_inner[2], flux_inner[3], flux_inner[4],
                   normal_energy_flux)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Isothermal})(flux_inner,
                                                                                       u_inner,
                                                                                       normal::AbstractVector,
                                                                                       x,
                                                                                       t,
                                                                                       operator_type::Gradient,
                                                                                       equations::CompressibleLES3D)
    v1, v2, v3 = boundary_condition.boundary_condition_velocity.boundary_value_function(x,
                                                                                        t,
                                                                                        equations)
    T = boundary_condition.boundary_condition_heat_flux.boundary_value_function(x, t,
                                                                                equations)
    return SVector(u_inner[1], v1, v2, v3, T)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Isothermal})(flux_inner,
                                                                                       u_inner,
                                                                                       normal::AbstractVector,
                                                                                       x,
                                                                                       t,
                                                                                       operator_type::Divergence,
                                                                                       equations::CompressibleLES3D)
    return flux_inner
end
end # @muladd
