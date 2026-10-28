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
    @unpack gamma, inv_gamma_minus_one = equations

    Pr = promote_type(typeof(gamma), typeof(Prandtl), typeof(R))(Prandtl)
    R = convert(typeof(Pr), R)
    # Under the assumption of constant Prandtl number the thermal conductivity
    # constant is kappa = gamma μ * R / ((gamma-1) Prandtl), see
    # `CompressibleNavierStokesDiffusion3D`
    kappa_over_mu = gamma * inv_gamma_minus_one * R / Pr

    gamma_over_Pr = gamma / Pr
    max_visc_cond = max(4 / 3, gamma_over_Pr)

    return CompressibleLES3D{typeof(Pr), typeof(mu), typeof(equations),
                             typeof(subgrid_model)}(R, mu, Pr, kappa_over_mu,
                                                    max_visc_cond, equations,
                                                    subgrid_model)
end

function Base.similar(equations::CompressibleLES3D, ::Type{NewRealT}) where {NewRealT}
    mu = equations.mu isa Real ? convert(NewRealT, equations.mu) : equations.mu
    return CompressibleLES3D(similar(equations.equations_hyperbolic, NewRealT); mu,
                             Prandtl = convert(NewRealT, equations.Pr),
                             R = convert(NewRealT, equations.R),
                             subgrid_model = similar(equations.subgrid_model, NewRealT))
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
@inline function cons2prim(u, equations::CompressibleLES3D)
    return cons2prim(u, equations.equations_hyperbolic)
end
@inline function prim2cons(u, equations::CompressibleLES3D)
    return prim2cons(u, equations.equations_hyperbolic)
end

function varnames(variable_mapping, equations::CompressibleLES3D)
    return varnames(variable_mapping, equations.equations_hyperbolic)
end

gradient_variable_transformation(::CompressibleLES3D) = cons2prim_temperature

# Primitive variables `(rho, v1, v2, v3, T)` of the conservative variables `u`, see
# `CompressibleNavierStokesDiffusion3D`
@inline function cons2prim_temperature(u, equations::CompressibleLES3D)
    rho, rho_v1, rho_v2, rho_v3, _ = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    T = temperature(u, equations)

    return SVector(rho, v1, v2, v3, T)
end

@inline function temperature(u, equations::CompressibleLES3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e_total = u
    @unpack gamma, R = equations

    p = (gamma - 1) *
        (rho_e_total - 0.5f0 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho)
    T = p / (rho * R)
    return T
end

# Largest eigenvalue of the molecular diffusivity matrix, see
# `CompressibleNavierStokesDiffusion3D`
@inline function max_diffusivity(u, equations::CompressibleLES3D)
    return dynamic_viscosity_conservative(u, equations) / u[1] * equations.max_visc_cond
end

@inline function vorticity(u, gradients, equations::CompressibleLES3D)
    _, _, dv2dx, dv3dx, _ = gradients[1]
    _, dv1dy, _, dv3dy, _ = gradients[2]
    _, dv1dz, dv2dz, _, _ = gradients[3]

    return SVector(dv3dy - dv2dz, dv1dz - dv3dx, dv2dx - dv1dy)
end

@inline function enstrophy(u, gradients, equations::CompressibleLES3D)
    # Enstrophy is 0.5 rho ω⋅ω where ω = ∇ × v
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
    _, dv1dx, dv2dx, dv3dx, _ = gradients[1]
    _, dv1dy, dv2dy, dv3dy, _ = gradients[2]
    _, dv1dz, dv2dz, dv3dz, _ = gradients[3]
    dv_dx = SVector(dv1dx, dv2dx, dv3dx)
    dv_dy = SVector(dv1dy, dv2dy, dv3dy)
    dv_dz = SVector(dv1dz, dv2dz, dv3dz)
    return dv_dx, dv_dy, dv_dz
end

# Viscosity `mu + mu_t` and thermal conductivity `kappa + kappa_t` with the eddy
# conductivity `kappa_t = c_p mu_t / Prandtl_turbulent`, where `kappa_over_mu * Pr = c_p`
@inline function diffusivities(u, mu_t, equations::CompressibleLES3D)
    (; Pr, kappa_over_mu) = equations
    mu = dynamic_viscosity(u, equations)
    kappa = kappa_over_mu * (mu + Pr / equations.subgrid_model.Prandtl_turbulent * mu_t)
    return SVector(mu + mu_t, kappa)
end

# Parabolic flux of the compressible Navier-Stokes equations, see
# `CompressibleNavierStokesDiffusion3D`, with the given viscosity `mu` and thermal
# conductivity `kappa`. Since `kappa` is not proportional to `mu` with the eddy viscosity,
# the heat flux is not multiplied by `mu`.
@inline function flux(u, gradients, orientation::Integer, diffusivities,
                      equations::CompressibleLES3D)
    mu, kappa = diffusivities
    v1, v2, v3 = convert_transformed_to_velocity(u, equations)
    # Here `gradients` contains the gradients of the primitive variables
    # (rho, v1, v2, v3, T)
    _, dv1dx, dv2dx, dv3dx, dTdx = gradients[1]
    _, dv1dy, dv2dy, dv3dy, dTdy = gradients[2]
    _, dv1dz, dv2dz, dv3dz, dTdz = gradients[3]

    # Components of viscous stress tensor

    # Diagonal parts
    # (4 * (v1)_x / 3 - 2 * ((v2)_y + (v3)_z)) / 3)
    tau_11 = (4 * dv1dx - 2 * (dv2dy + dv3dz)) / 3
    # (4 * (v2)_y / 3 - 2 * ((v1)_x + (v3)_z) / 3)
    tau_22 = (4 * dv2dy - 2 * (dv1dx + dv3dz)) / 3
    # (4 * (v3)_z / 3 - 2 * ((v1)_x + (v2)_y) / 3)
    tau_33 = (4 * dv3dz - 2 * (dv1dx + dv2dy)) / 3

    # Off diagonal parts, exploit that stress tensor is symmetric
    # ((v1)_y + (v2)_x)
    tau_12 = dv1dy + dv2dx # = tau_21
    # ((v1)_z + (v3)_x)
    tau_13 = dv1dz + dv3dx # = tau_31
    # ((v2)_z + (v3)_y)
    tau_23 = dv2dz + dv3dy # = tau_32

    # Fourier's law q = -kappa * grad(T)
    q1 = kappa * dTdx
    q2 = kappa * dTdy
    q3 = kappa * dTdz

    if orientation == 1
        # parabolic flux components in the x-direction
        f1 = 0
        f2 = tau_11 * mu
        f3 = tau_12 * mu
        f4 = tau_13 * mu
        f5 = (v1 * tau_11 + v2 * tau_12 + v3 * tau_13) * mu + q1

        return SVector(f1, f2, f3, f4, f5)
    elseif orientation == 2
        # parabolic flux components in the y-direction
        # Note, symmetry is exploited for tau_12 = tau_21
        g1 = 0
        g2 = tau_12 * mu # tau_21 * mu
        g3 = tau_22 * mu
        g4 = tau_23 * mu
        g5 = (v1 * tau_12 + v2 * tau_22 + v3 * tau_23) * mu + q2

        return SVector(g1, g2, g3, g4, g5)
    else # if orientation == 3
        # parabolic flux components in the z-direction
        # Note, symmetry is exploited for tau_13 = tau_31, tau_23 = tau_32
        h1 = 0
        h2 = tau_13 * mu # tau_31 * mu
        h3 = tau_23 * mu # tau_32 * mu
        h4 = tau_33 * mu
        h5 = (v1 * tau_13 + v2 * tau_23 + v3 * tau_33) * mu + q3

        return SVector(h1, h2, h3, h4, h5)
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
    _, tau_1n, tau_2n, tau_3n, _ = flux_inner # extract fluxes for 2nd, 3rd, and 4th equations
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
