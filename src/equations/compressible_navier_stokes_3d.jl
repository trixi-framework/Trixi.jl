# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    CompressibleNavierStokesDiffusion3D(equations; mu, Pr,
                                        gradient_variables=GradientVariablesPrimitive())

Contains the diffusion (i.e. parabolic) terms applied
to mass, momenta, and total energy together with the advective terms from
the [`CompressibleEulerEquations3D`](@ref).

- `equations`: instance of the [`CompressibleEulerEquations3D`](@ref)
- `mu`: dynamic viscosity,
- `Pr`: Prandtl number,
- `gradient_variables`: which variables the gradients are taken with respect to.
                        Defaults to [`GradientVariablesPrimitive()`](@ref).
                        For an entropy stable formulation, use [`GradientVariablesEntropy()`](@ref).

Fluid properties such as the dynamic viscosity ``\mu`` can be provided in any consistent unit system, e.g.,
[``\mu``] = kg m⁻¹ s⁻¹.
The viscosity ``\mu`` may be a constant or a function of the current state, e.g., 
depending on temperature (Sutherland's law): ``\mu = \mu(T)``.
In the latter case, the function `mu` needs to have the signature `mu(u, equations)`.

The particular form of the compressible Navier-Stokes implemented is
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
\rho \\ \rho \mathbf{v} \\ \rho e
\end{pmatrix}
+
\nabla \cdot
\begin{pmatrix}
 \rho \mathbf{v} \\ \rho \mathbf{v}\mathbf{v}^T + p \underline{I} \\ (\rho e + p) \mathbf{v}
\end{pmatrix}
=
\nabla \cdot
\begin{pmatrix}
0 \\ \underline{\tau} \\ \underline{\tau}\mathbf{v} - \mathbf{q}
\end{pmatrix}
```
where the system is closed with the ideal gas assumption giving
```math
p = (\gamma - 1) \left( \rho e - \frac{1}{2} \rho (v_1^2+v_2^2+v_3^2) \right)
```
as the pressure. The value of the adiabatic constant `gamma` is taken from the [`CompressibleEulerEquations2D`](@ref).
The terms on the right hand side of the system above
are built from the viscous stress tensor
```math
\underline{\tau} = \mu \left(\nabla\mathbf{v} + \left(\nabla\mathbf{v}\right)^T\right) - \frac{2}{3} \mu \left(\nabla\cdot\mathbf{v}\right)\underline{I}
```
where ``\underline{I}`` is the ``3\times 3`` identity matrix and the heat flux is
```math
\mathbf{q} = -\kappa\nabla\left(T\right),\quad T = \frac{p}{R\rho}
```
where ``T`` is the temperature and ``\kappa`` is the thermal conductivity for Fick's law.
Under the assumption that the gas has a constant Prandtl number,
the thermal conductivity is
```math
\kappa = \frac{\gamma \mu R}{(\gamma - 1)\textrm{Pr}}.
```
From this combination of temperature ``T`` and thermal conductivity ``\kappa`` we see
that the gas constant `R` cancels and the heat flux becomes
```math
\mathbf{q} = -\kappa\nabla\left(T\right) = -\frac{\gamma \mu}{(\gamma - 1)\textrm{Pr}}\nabla\left(\frac{p}{\rho}\right)
```
which is the form implemented below in the [`flux`](@ref) function.

In two spatial dimensions we require gradients for three quantities, e.g.,
primitive quantities
```math
\nabla v_1,\, \nabla v_2,\, \nabla v_3,\, \nabla T
```
or the entropy variables
```math
\nabla w_2,\, \nabla w_3,\, \nabla w_4\, \nabla w_5
```
where
```math
w_2 = \frac{\rho v_1}{p},\, w_3 = \frac{\rho v_2}{p},\, w_4 = \frac{\rho v_3}{p},\, w_5 = -\frac{\rho}{p}
```
"""
struct CompressibleNavierStokesDiffusion3D{GradientVariables, RealT <: Real, Mu,
                                           E <: AbstractCompressibleEulerEquations{3}} <:
       AbstractCompressibleNavierStokesDiffusion{3, 5, GradientVariables}
    # TODO: parabolic
    # 1) For now save gamma and inv(gamma-1) again, but could potentially reuse them from the Euler equations
    # 2) Add NGRADS as a type parameter here and in AbstractEquationsParabolic, add `ngradients(...)` accessor function
    gamma::RealT               # ratio of specific heats
    inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications

    mu::Mu                     # viscosity
    Pr::RealT                  # Prandtl number
    kappa::RealT               # thermal diffusivity for Fick's law

    equations_hyperbolic::E    # CompressibleEulerEquations3D
    gradient_variables::GradientVariables # GradientVariablesPrimitive or GradientVariablesEntropy
end

# default to primitive gradient variables
function CompressibleNavierStokesDiffusion3D(equations::CompressibleEulerEquations3D;
                                             mu, Prandtl,
                                             gradient_variables = GradientVariablesPrimitive())
    gamma = equations.gamma
    inv_gamma_minus_one = equations.inv_gamma_minus_one

    # Under the assumption of constant Prandtl number the thermal conductivity
    # constant is kappa = gamma μ / ((gamma-1) Prandtl).
    # Important note! Factor of μ is accounted for later in `flux`.
    # This avoids recomputation of kappa for non-constant μ.
    kappa = gamma * inv_gamma_minus_one / Prandtl

    CompressibleNavierStokesDiffusion3D{typeof(gradient_variables), typeof(gamma),
                                        typeof(mu),
                                        typeof(equations)}(gamma, inv_gamma_minus_one,
                                                           mu, Prandtl, kappa,
                                                           equations,
                                                           gradient_variables)
end

# TODO: parabolic
# This is the flexibility a user should have to select the different gradient variable types
# varnames(::typeof(cons2prim)   , ::CompressibleNavierStokesDiffusion3D) = ("v1", "v2", "v3", "T")
# varnames(::typeof(cons2entropy), ::CompressibleNavierStokesDiffusion3D) = ("w2", "w3", "w4", "w5")

function varnames(variable_mapping,
                  equations_parabolic::CompressibleNavierStokesDiffusion3D)
    varnames(variable_mapping, equations_parabolic.equations_hyperbolic)
end

# we specialize this function to compute gradients of primitive variables instead of
# conservative variables.
function gradient_variable_transformation(::CompressibleNavierStokesDiffusion3D{GradientVariablesPrimitive})
    cons2prim
end
function gradient_variable_transformation(::CompressibleNavierStokesDiffusion3D{GradientVariablesEntropy})
    cons2entropy
end

# Explicit formulas for the diffusive Navier-Stokes fluxes are available, e.g., in Section 2
# of the paper by Rueda-Ramírez, Hennemann, Hindenlang, Winters, and Gassner
# "An Entropy Stable Nodal Discontinuous Galerkin Method for the resistive
#  MHD Equations. Part II: Subcell Finite Volume Shock Capturing"
# where one sets the magnetic field components equal to 0.
function flux(u, gradients, orientation::Integer,
              equations::CompressibleNavierStokesDiffusion3D)
    # Here, `u` is assumed to be the "transformed" variables specified by `gradient_variable_transformation`.
    _, v1, v2, v3, _ = convert_transformed_to_primitive(u, equations)
    # Here `gradients` is assumed to contain the gradients of the primitive variables (rho, v1, v2, v3, T)
    # either computed directly or reverse engineered from the gradient of the entropy variables
    # by way of the `convert_gradient_variables` function.
    _, dv1dx, dv2dx, dv3dx, dTdx = convert_derivative_to_primitive(u, gradients[1],
                                                                   equations)
    _, dv1dy, dv2dy, dv3dy, dTdy = convert_derivative_to_primitive(u, gradients[2],
                                                                   equations)
    _, dv1dz, dv2dz, dv3dz, dTdz = convert_derivative_to_primitive(u, gradients[3],
                                                                   equations)

    # Components of viscous stress tensor

    # Diagonal parts
    # (4 * (v1)_x / 3 - 2 * ((v2)_y + (v3)_z)) / 3)
    tau_11 = 4 * dv1dx / 3 - 2 * (dv2dy + dv3dz) / 3
    # (4 * (v2)_y / 3 - 2 * ((v1)_x + (v3)_z) / 3)
    tau_22 = 4 * dv2dy / 3 - 2 * (dv1dx + dv3dz) / 3
    # (4 * (v3)_z / 3 - 2 * ((v1)_x + (v2)_y) / 3)
    tau_33 = 4 * dv3dz / 3 - 2 * (dv1dx + dv2dy) / 3

    # Off diagonal parts, exploit that stress tensor is symmetric
    # ((v1)_y + (v2)_x)
    tau_12 = dv1dy + dv2dx # = tau_21
    # ((v1)_z + (v3)_x)
    tau_13 = dv1dz + dv3dx # = tau_31
    # ((v2)_z + (v3)_y)
    tau_23 = dv2dz + dv3dy # = tau_32

    # Fick's law q = -kappa * grad(T) = -kappa * grad(p / (R rho))
    # with thermal diffusivity constant kappa = gamma μ R / ((gamma-1) Pr)
    # Note, the gas constant cancels under this formulation, so it is not present
    # in the implementation
    q1 = equations.kappa * dTdx
    q2 = equations.kappa * dTdy
    q3 = equations.kappa * dTdz

    # In the simplest cases, the user passed in `mu` or `mu()` 
    # (which returns just a constant) but
    # more complex functions like Sutherland's law are possible.
    # `dynamic_viscosity` is a helper function that handles both cases
    # by dispatching on the type of `equations.mu`.
    mu = dynamic_viscosity(u, equations)

    if orientation == 1
        # viscous flux components in the x-direction
        f1 = 0
        f2 = tau_11 * mu
        f3 = tau_12 * mu
        f4 = tau_13 * mu
        f5 = (v1 * tau_11 + v2 * tau_12 + v3 * tau_13 + q1) * mu

        return SVector(f1, f2, f3, f4, f5)
    elseif orientation == 2
        # viscous flux components in the y-direction
        # Note, symmetry is exploited for tau_12 = tau_21
        g1 = 0
        g2 = tau_12 * mu # tau_21 * mu
        g3 = tau_22 * mu
        g4 = tau_23 * mu
        g5 = (v1 * tau_12 + v2 * tau_22 + v3 * tau_23 + q2) * mu

        return SVector(g1, g2, g3, g4, g5)
    else # if orientation == 3
        # viscous flux components in the z-direction
        # Note, symmetry is exploited for tau_13 = tau_31, tau_23 = tau_32
        h1 = 0
        h2 = tau_13 * mu # tau_31 * mu
        h3 = tau_23 * mu # tau_32 * mu
        h4 = tau_33 * mu
        h5 = (v1 * tau_13 + v2 * tau_23 + v3 * tau_33 + q3) * mu

        return SVector(h1, h2, h3, h4, h5)
    end
end

# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleNavierStokesDiffusion3D)
    rho, rho_v1, rho_v2, rho_v3, _ = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    T = temperature(u, equations)

    return SVector(rho, v1, v2, v3, T)
end

# Convert conservative variables to entropy
# TODO: parabolic. We can improve efficiency by not computing w_1, which involves logarithms
# This can be done by specializing `cons2entropy` and `entropy2cons` to `CompressibleNavierStokesDiffusion2D`,
# but this may be confusing to new users.
function cons2entropy(u, equations::CompressibleNavierStokesDiffusion3D)
    cons2entropy(u, equations.equations_hyperbolic)
end
function entropy2cons(w, equations::CompressibleNavierStokesDiffusion3D)
    entropy2cons(w, equations.equations_hyperbolic)
end

# the `flux` function takes in transformed variables `u` which depend on the type of the gradient variables.
# For CNS, it is simplest to formulate the viscous terms in primitive variables, so we transform the transformed
# variables into primitive variables.
@inline function convert_transformed_to_primitive(u_transformed,
                                                  equations::CompressibleNavierStokesDiffusion3D{GradientVariablesPrimitive})
    return u_transformed
end

# TODO: parabolic. Make this more efficient!
@inline function convert_transformed_to_primitive(u_transformed,
                                                  equations::CompressibleNavierStokesDiffusion3D{GradientVariablesEntropy})
    # note: this uses CompressibleNavierStokesDiffusion3D versions of cons2prim and entropy2cons
    return cons2prim(entropy2cons(u_transformed, equations), equations)
end

# Takes the solution values `u` and gradient of the entropy variables (w_2, w_3, w_4, w_5) and
# reverse engineers the gradients to be terms of the primitive variables (v1, v2, v3, T).
# Helpful because then the diffusive fluxes have the same form as on paper.
# Note, the first component of `gradient_entropy_vars` contains gradient(rho) which is unused.
# TODO: parabolic; entropy stable viscous terms
@inline function convert_derivative_to_primitive(u, gradient,
                                                 ::CompressibleNavierStokesDiffusion3D{GradientVariablesPrimitive})
    return gradient
end

# the first argument is always the "transformed" variables.
@inline function convert_derivative_to_primitive(w, gradient_entropy_vars,
                                                 equations::CompressibleNavierStokesDiffusion3D{GradientVariablesEntropy})

    # TODO: parabolic. This is inefficient to pass in transformed variables but then transform them back.
    # We can fix this if we directly compute v1, v2, v3, T from the entropy variables
    u = entropy2cons(w, equations) # calls a "modified" entropy2cons defined for CompressibleNavierStokesDiffusion3D
    rho, rho_v1, rho_v2, rho_v3, _ = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    T = temperature(u, equations)

    return SVector(gradient_entropy_vars[1],
                   T * (gradient_entropy_vars[2] + v1 * gradient_entropy_vars[5]), # grad(u) = T*(grad(w_2)+v1*grad(w_5))
                   T * (gradient_entropy_vars[3] + v2 * gradient_entropy_vars[5]), # grad(v) = T*(grad(w_3)+v2*grad(w_5))
                   T * (gradient_entropy_vars[4] + v3 * gradient_entropy_vars[5]), # grad(v) = T*(grad(w_4)+v3*grad(w_5))
                   T * T * gradient_entropy_vars[5])
end

# This routine is required because `prim2cons` is called in `initial_condition`, which
# is called with `equations::CompressibleEulerEquations3D`. This means it is inconsistent
# with `cons2prim(..., ::CompressibleNavierStokesDiffusion3D)` as defined above.
# TODO: parabolic. Is there a way to clean this up?
@inline function prim2cons(u, equations::CompressibleNavierStokesDiffusion3D)
    prim2cons(u, equations.equations_hyperbolic)
end

@inline function temperature(u, equations::CompressibleNavierStokesDiffusion3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e = u

    p = (equations.gamma - 1) * (rho_e - 0.5f0 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho)
    T = p / rho
    return T
end

@inline function velocity(u, equations::CompressibleNavierStokesDiffusion3D)
    rho = u[1]
    v1 = u[2] / rho
    v2 = u[3] / rho
    v3 = u[4] / rho
    return SVector(v1, v2, v3)
end

@doc raw"""
    enstrophy(u, gradients, equations::CompressibleNavierStokesDiffusion3D)

Computes the (node-wise) enstrophy, defined as
```math
    \mathcal{E} = \frac{1}{2} \rho \boldsymbol{\omega} \cdot \boldsymbol{\omega}
```
where ``\boldsymbol{\omega} = \nabla \times \boldsymbol{v}`` is the [`vorticity`](@ref).
In 3D, ``\boldsymbol{\omega}`` is a full three-component vector.
"""
@inline function enstrophy(u, gradients, equations::CompressibleNavierStokesDiffusion3D)
    # Enstrophy is 0.5 rho ω⋅ω where ω = ∇ × v

    omega = vorticity(u, gradients, equations)
    return 0.5f0 * u[1] * (omega[1]^2 + omega[2]^2 + omega[3]^2)
end

@doc raw"""
    vorticity(u, gradients, equations::CompressibleNavierStokesDiffusion3D)

Computes the (node-wise) vorticity, defined in 3D as
```math
    \boldsymbol{\omega} = \nabla \times \boldsymbol{v} =
    \begin{pmatrix}
        \frac{\partial v_3}{\partial y} - \frac{\partial v_2}{\partial z} \\
        \frac{\partial v_1}{\partial z} - \frac{\partial v_3}{\partial x} \\
        \frac{\partial v_2}{\partial x} - \frac{\partial v_1}{\partial y}
    \end{pmatrix}
```
"""
@inline function vorticity(u, gradients, equations::CompressibleNavierStokesDiffusion3D)
    # Ensure that we have velocity `gradients` by way of the `convert_gradient_variables` function.
    _, _, dv2dx, dv3dx, _ = convert_derivative_to_primitive(u, gradients[1], equations)
    _, dv1dy, _, dv3dy, _ = convert_derivative_to_primitive(u, gradients[2], equations)
    _, dv1dz, dv2dz, _, _ = convert_derivative_to_primitive(u, gradients[3], equations)

    return SVector(dv3dy - dv2dz, dv1dz - dv3dx, dv2dx - dv1dy)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      u_inner,
                                                                                      normal::AbstractVector,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Gradient,
                                                                                      equations::CompressibleNavierStokesDiffusion3D{GradientVariablesPrimitive})
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
                                                                                      equations::CompressibleNavierStokesDiffusion3D{GradientVariablesPrimitive})
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
                                                                                       equations::CompressibleNavierStokesDiffusion3D{GradientVariablesPrimitive})
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
                                                                                       equations::CompressibleNavierStokesDiffusion3D{GradientVariablesPrimitive})
    return flux_inner
end

# specialized BC impositions for GradientVariablesEntropy.

# This should return a SVector containing the boundary values of entropy variables.
# Here, `w_inner` are the transformed variables (e.g., entropy variables).
#
# Taken from "Entropy stable modal discontinuous Galerkin schemes and wall boundary conditions
#             for the compressible Navier-Stokes equations" by Chan, Lin, Warburton 2022.
# DOI: 10.1016/j.jcp.2021.110723
@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      w_inner,
                                                                                      normal::AbstractVector,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Gradient,
                                                                                      equations::CompressibleNavierStokesDiffusion3D{GradientVariablesEntropy})
    v1, v2, v3 = boundary_condition.boundary_condition_velocity.boundary_value_function(x,
                                                                                        t,
                                                                                        equations)
    negative_rho_inv_p = w_inner[5] # w_5 = -rho / p
    return SVector(w_inner[1], -v1 * negative_rho_inv_p, -v2 * negative_rho_inv_p,
                   -v3 * negative_rho_inv_p, negative_rho_inv_p)
end

# this is actually identical to the specialization for GradientVariablesPrimitive, but included for completeness.
@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      w_inner,
                                                                                      normal::AbstractVector,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Divergence,
                                                                                      equations::CompressibleNavierStokesDiffusion3D{GradientVariablesEntropy})
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
                                                                                       w_inner,
                                                                                       normal::AbstractVector,
                                                                                       x,
                                                                                       t,
                                                                                       operator_type::Gradient,
                                                                                       equations::CompressibleNavierStokesDiffusion3D{GradientVariablesEntropy})
    v1, v2, v3 = boundary_condition.boundary_condition_velocity.boundary_value_function(x,
                                                                                        t,
                                                                                        equations)
    T = boundary_condition.boundary_condition_heat_flux.boundary_value_function(x, t,
                                                                                equations)

    # the entropy variables w2 = rho * v1 / p = v1 / T = -v1 * w5. Similarly for w3 and w4
    w5 = -1 / T
    return SVector(w_inner[1], -v1 * w5, -v2 * w5, -v3 * w5, w5)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Isothermal})(flux_inner,
                                                                                       w_inner,
                                                                                       normal::AbstractVector,
                                                                                       x,
                                                                                       t,
                                                                                       operator_type::Divergence,
                                                                                       equations::CompressibleNavierStokesDiffusion3D{GradientVariablesEntropy})
    return SVector(flux_inner[1], flux_inner[2], flux_inner[3], flux_inner[4],
                   flux_inner[5])
end

# Computes the mirror velocity across a symmetry plane which enforces
# a tangential velocity that is aligned with the symmetry plane, i.e.,
# which is normal to the `normal_direction`.
# See also `boundary_condition_slip_wall` and `rotate_to_x`.
@inline function velocity_symmetry_plane(normal_direction::AbstractVector, v1, v2, v3)
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    # Compute alignment of velocity with normal direction
    v_normal = v1 * normal[1] + v2 * normal[2] + v3 * normal[3]

    v1_outer = v1 - 2 * v_normal * normal[1]
    v2_outer = v2 - 2 * v_normal * normal[2]
    v3_outer = v3 - 2 * v_normal * normal[3]

    return v1_outer, v2_outer, v3_outer
end

# Note: This should be used with `boundary_condition_slip_wall` for the hyperbolic (Euler) part.
@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:Slip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      u_inner,
                                                                                      normal::AbstractVector,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Gradient,
                                                                                      equations::CompressibleNavierStokesDiffusion3D{GradientVariablesPrimitive})
    v1_outer, v2_outer, v3_outer = velocity_symmetry_plane(normal,
                                                           u_inner[2],
                                                           u_inner[3],
                                                           u_inner[4])

    return SVector(u_inner[1], v1_outer, v2_outer, v3_outer, u_inner[5])
end

# Note: This should be used with `boundary_condition_slip_wall` for the hyperbolic (Euler) part.
@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:Slip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      u_inner,
                                                                                      normal::AbstractVector,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Divergence,
                                                                                      equations::CompressibleNavierStokesDiffusion3D{GradientVariablesPrimitive})
    normal_heat_flux = boundary_condition.boundary_condition_heat_flux.boundary_value_normal_flux_function(x,
                                                                                                           t,
                                                                                                           equations)
    # Normal stresses should be 0. This implies also that `normal_energy_flux = normal_heat_flux`.
    # For details, see Section 4.2 of 
    # "Entropy stable modal discontinuous Galerkin schemes and wall boundary conditions
    #  for the compressible Navier-Stokes equations" by Chan, Lin, Warburton 2022.
    # DOI: 10.1016/j.jcp.2021.110723
    return SVector(flux_inner[1], 0, 0, 0, normal_heat_flux)
end

# Dirichlet Boundary Condition for e.g. P4est mesh
@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner,
                                                                  u_inner,
                                                                  normal::AbstractVector,
                                                                  x, t,
                                                                  operator_type::Gradient,
                                                                  equations::CompressibleNavierStokesDiffusion3D{GradientVariablesPrimitive})
    # BCs are usually specified as conservative variables so we convert them to primitive variables
    #  because the gradients are assumed to be with respect to the primitive variables
    u_boundary = boundary_condition.boundary_value_function(x, t, equations)

    return cons2prim(u_boundary, equations)
end

@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner,
                                                                  u_inner,
                                                                  normal::AbstractVector,
                                                                  x, t,
                                                                  operator_type::Divergence,
                                                                  equations::CompressibleNavierStokesDiffusion3D{GradientVariablesPrimitive})
    # for Dirichlet boundary conditions, we do not impose any conditions on the viscous fluxes
    return flux_inner
end
end # @muladd
