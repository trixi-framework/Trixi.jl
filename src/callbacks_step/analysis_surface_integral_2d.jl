# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LiftCoefficientPressure2D(aoa, rho_inf, u_inf, l_inf)

Compute the lift coefficient
```math
C_{L,p} \coloneqq \frac{\oint_{\partial \Omega} p \boldsymbol n \cdot \psi_L \, \mathrm{d} S}
                        {0.5 \rho_{\infty} U_{\infty}^2 L_{\infty}}
```
based on the pressure distribution along a boundary.
In 2D, the freestream-normal unit vector ``\psi_L`` is given by
```math
\psi_L \coloneqq \begin{pmatrix} -\sin(\alpha) \\ \cos(\alpha) \end{pmatrix}
```
where ``\alpha`` is the angle of attack.
Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the the to-be-computed variables (for instance `LiftCoefficientPressure2D`) 
and boundary information.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rho_inf::Real`: Free-stream density
- `u_inf::Real`: Free-stream velocity
- `l_inf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function LiftCoefficientPressure2D(aoa, rho_inf, u_inf, l_inf)
    # `psi_lift` is the normal unit vector to the freestream direction.
    # Note: The choice of the normal vector `psi_lift = (-sin(aoa), cos(aoa))`
    # leads to positive lift coefficients for positive angles of attack for airfoils.
    # One could also use `psi_lift = (sin(aoa), -cos(aoa))` which results in the same
    # value, but with the opposite sign.
    psi_lift = (-sin(aoa), cos(aoa))
    return LiftCoefficientPressure(ForceState(psi_lift, rho_inf, u_inf, l_inf))
end

@doc raw"""
    DragCoefficientPressure2D(aoa, rho_inf, u_inf, l_inf)

Compute the drag coefficient
```math
C_{D,p} \coloneqq \frac{\oint_{\partial \Omega} p \boldsymbol n \cdot \psi_D \, \mathrm{d} S}
                        {0.5 \rho_{\infty} U_{\infty}^2 L_{\infty}}
```
based on the pressure distribution along a boundary.
In 2D, the freestream-tangent unit vector ``\psi_D`` is given by
```math
\psi_D \coloneqq \begin{pmatrix} \cos(\alpha) \\ \sin(\alpha) \end{pmatrix}
```
where ``\alpha`` is the angle of attack.

Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the the to-be-computed variables (for instance `DragCoefficientPressure2D`) 
and boundary information.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rho_inf::Real`: Free-stream density
- `u_inf::Real`: Free-stream velocity
- `l_inf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function DragCoefficientPressure2D(aoa, rho_inf, u_inf, l_inf)
    # `psi_drag` is the unit vector tangent to the freestream direction
    psi_drag = (cos(aoa), sin(aoa))
    return DragCoefficientPressure(ForceState(psi_drag, rho_inf, u_inf, l_inf))
end

@doc raw"""
    LiftCoefficientShearStress2D(aoa, rho_inf, u_inf, l_inf)

Compute the lift coefficient
```math
C_{L,f} \coloneqq \frac{\oint_{\partial \Omega} \boldsymbol \tau_w \cdot \psi_L \, \mathrm{d} S}
                        {0.5 \rho_{\infty} U_{\infty}^2 L_{\infty}}
```
based on the wall shear stress vector ``\tau_w`` along a boundary.
In 2D, the freestream-normal unit vector ``\psi_L`` is given by
```math
\psi_L \coloneqq \begin{pmatrix} -\sin(\alpha) \\ \cos(\alpha) \end{pmatrix}
```
where ``\alpha`` is the angle of attack.
Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the the to-be-computed variables (for instance `LiftCoefficientShearStress2D`) 
and boundary information.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rho_inf::Real`: Free-stream density
- `u_inf::Real`: Free-stream velocity
- `l_inf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function LiftCoefficientShearStress2D(aoa, rho_inf, u_inf, l_inf)
    # `psi_lift` is the normal unit vector to the freestream direction.
    # Note: The choice of the normal vector `psi_lift = (-sin(aoa), cos(aoa))`
    # leads to negative lift coefficients for airfoils.
    # One could also use `psi_lift = (sin(aoa), -cos(aoa))` which results in the same
    # value, but with the opposite sign.
    psi_lift = (-sin(aoa), cos(aoa))
    return LiftCoefficientShearStress(ForceState(psi_lift, rho_inf, u_inf, l_inf))
end

@doc raw"""
    DragCoefficientShearStress2D(aoa, rho_inf, u_inf, l_inf)

Compute the drag coefficient
```math
C_{D,f} \coloneqq \frac{\oint_{\partial \Omega} \boldsymbol \tau_w \cdot \psi_D \, \mathrm{d} S}
                        {0.5 \rho_{\infty} U_{\infty}^2 L_{\infty}}
```
based on the wall shear stress vector ``\tau_w`` along a boundary.
In 2D, the freestream-tangent unit vector ``\psi_D`` is given by
```math
\psi_D \coloneqq \begin{pmatrix} \cos(\alpha) \\ \sin(\alpha) \end{pmatrix}
```
where ``\alpha`` is the angle of attack.
Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the the to-be-computed variables (for instance `DragCoefficientShearStress2D`) 
and boundary information.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rho_inf::Real`: Free-stream density
- `u_inf::Real`: Free-stream velocity
- `l_inf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function DragCoefficientShearStress2D(aoa, rho_inf, u_inf, l_inf)
    # `psi_drag` is the unit vector tangent to the freestream direction
    psi_drag = (cos(aoa), sin(aoa))
    return DragCoefficientShearStress(ForceState(psi_drag, rho_inf, u_inf, l_inf))
end

# Compute the three components of the 2D symmetric viscous stress tensor
# (tau_11, tau_12, tau_22) based on the gradients of the velocity field.
# This is required for drag and lift coefficients based on shear stress,
# as well as for the non-integrated quantities such as
# skin friction coefficient (to be added).
function viscous_stress_tensor(u, normal_direction, equations_parabolic,
                               gradients_1, gradients_2)
    _, dv1dx, dv2dx, _ = convert_derivative_to_primitive(u, gradients_1,
                                                         equations_parabolic)
    _, dv1dy, dv2dy, _ = convert_derivative_to_primitive(u, gradients_2,
                                                         equations_parabolic)

    # Components of viscous stress tensor
    # (4/3 * (v1)_x - 2/3 * (v2)_y)
    tau_11 = 4.0 / 3.0 * dv1dx - 2.0 / 3.0 * dv2dy
    # ((v1)_y + (v2)_x)
    # stress tensor is symmetric
    tau_12 = dv1dy + dv2dx # = tau_21
    # (4/3 * (v2)_y - 2/3 * (v1)_x)
    tau_22 = 4.0 / 3.0 * dv2dy - 2.0 / 3.0 * dv1dx

    mu = dynamic_viscosity(u, equations_parabolic)

    return mu .* (tau_11, tau_12, tau_22)
end

# 2D viscous stress vector based on contracting the viscous stress tensor
# with the normalized `normal_direction` vector.
function viscous_stress_vector(u, normal_direction, equations_parabolic,
                               gradients_1, gradients_2)
    #  Normalize normal direction, should point *into* the fluid => *(-1)
    n_normal = -normal_direction / norm(normal_direction)

    tau_11, tau_12, tau_22 = viscous_stress_tensor(u, normal_direction,
                                                   equations_parabolic,
                                                   gradients_1, gradients_2)

    # Viscous stress vector: Stress tensor * normal vector
    viscous_stress_vector_1 = tau_11 * n_normal[1] + tau_12 * n_normal[2]
    viscous_stress_vector_2 = tau_12 * n_normal[1] + tau_22 * n_normal[2]

    return (viscous_stress_vector_1, viscous_stress_vector_2)
end

function (lift_coefficient::LiftCoefficientShearStress{RealT, 2})(u, normal_direction,
                                                                  x, t,
                                                                  equations_parabolic,
                                                                  gradients_1,
                                                                  gradients_2) where {RealT <:
                                                                                      Real}
    visc_stress_vector = viscous_stress_vector(u, normal_direction, equations_parabolic,
                                               gradients_1, gradients_2)
    @unpack psi, rho_inf, u_inf, l_inf = lift_coefficient.force_state
    return (visc_stress_vector[1] * psi[1] + visc_stress_vector[2] * psi[2]) /
           (0.5f0 * rho_inf * u_inf^2 * l_inf)
end

function (drag_coefficient::DragCoefficientShearStress{RealT, 2})(u, normal_direction,
                                                                  x, t,
                                                                  equations_parabolic,
                                                                  gradients_1,
                                                                  gradients_2) where {RealT <:
                                                                                      Real}
    visc_stress_vector = viscous_stress_vector(u, normal_direction, equations_parabolic,
                                               gradients_1, gradients_2)
    @unpack psi, rho_inf, u_inf, l_inf = drag_coefficient.force_state
    return (visc_stress_vector[1] * psi[1] + visc_stress_vector[2] * psi[2]) /
           (0.5f0 * rho_inf * u_inf^2 * l_inf)
end

# 2D version of the `analyze` function for `AnalysisSurfaceIntegral`, i.e., 
# `LiftCoefficientPressure` and `DragCoefficientPressure`.
function analyze(surface_variable::AnalysisSurfaceIntegral, du, u, t,
                 mesh::P4estMesh{2},
                 equations, dg::DGSEM, cache, semi)
    @unpack boundaries = cache
    @unpack node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack variable, boundary_symbols = surface_variable
    @unpack boundary_symbol_indices = semi.boundary_conditions
    boundary_indices = get_boundary_indices(boundary_symbols, boundary_symbol_indices)

    surface_integral = zero(eltype(u))
    index_range = eachnode(dg)
    for boundary in boundary_indices
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in index_range
            u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg,
                                         node_index, boundary)
            # Extract normal direction at nodes which points from the elements outwards,
            # i.e., *into* the structure.
            normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node, element)

            # Coordinates at a boundary node
            x = get_node_coords(node_coordinates, equations, dg,
                                i_node, j_node, element)

            # L2 norm of normal direction (contravariant_vector) is the surface element
            dS = weights[node_index] * norm(normal_direction)

            # Integral over entire boundary surface. Note, it is assumed that the
            # `normal_direction` is normalized to be a normal vector within the
            # function `variable` and the division of the normal scaling factor
            # `norm(normal_direction)` is then accounted for with the `dS` quantity.
            surface_integral += variable(u_node, normal_direction, x, t, equations) * dS

            i_node += i_node_step
            j_node += j_node_step
        end
    end
    return surface_integral
end

# 2D version of the `analyze` function for `AnalysisSurfaceIntegral` of viscous, i.e.,
# variables that require gradients of the solution variables.
# These are for parabolic equations readily available.
# Examples are `LiftCoefficientShearStress` and `DragCoefficientShearStress`.
function analyze(surface_variable::AnalysisSurfaceIntegral{Variable}, du, u, t,
                 mesh::P4estMesh{2},
                 equations, equations_parabolic,
                 dg::DGSEM, cache, semi,
                 cache_parabolic) where {Variable <: VariableViscous}
    @unpack boundaries = cache
    @unpack node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack variable, boundary_symbols = surface_variable
    @unpack boundary_symbol_indices = semi.boundary_conditions
    boundary_indices = get_boundary_indices(boundary_symbols, boundary_symbol_indices)

    # Additions for parabolic
    @unpack viscous_container = cache_parabolic
    @unpack gradients = viscous_container

    gradients_x, gradients_y = gradients

    surface_integral = zero(eltype(u))
    index_range = eachnode(dg)
    for boundary in boundary_indices
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in index_range
            u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg,
                                         node_index, boundary)
            # Extract normal direction at nodes which points from the elements outwards,
            # i.e., *into* the structure.
            normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node, element)

            # Coordinates at a boundary node
            x = get_node_coords(node_coordinates, equations, dg,
                                i_node, j_node, element)

            # L2 norm of normal direction (contravariant_vector) is the surface element
            dS = weights[node_index] * norm(normal_direction)

            gradients_1 = get_node_vars(gradients_x, equations_parabolic, dg,
                                        i_node, j_node, element)
            gradients_2 = get_node_vars(gradients_y, equations_parabolic, dg,
                                        i_node, j_node, element)

            # Integral over whole boundary surface. Note, it is assumed that the
            # `normal_direction` is normalized to be a normal vector within the
            # function `variable` and the division of the normal scaling factor
            # `norm(normal_direction)` is then accounted for with the `dS` quantity.
            surface_integral += variable(u_node, normal_direction, x, t,
                                         equations_parabolic,
                                         gradients_1, gradients_2) * dS

            i_node += i_node_step
            j_node += j_node_step
        end
    end
    return surface_integral
end
end # muladd
