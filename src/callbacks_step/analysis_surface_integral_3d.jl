# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LiftCoefficientPressure3D(aoa, rho_inf, u_inf, a_inf)

Compute the lift coefficient
```math
C_{L,p} \coloneqq \frac{\oint_{\partial \Omega} p \boldsymbol n \cdot \psi_L \, \mathrm{d} S}
                        {0.5 \rho_{\infty} U_{\infty}^2 A_{\infty}}
```
based on the pressure distribution along a boundary.
In 3D, the freestream-normal unit vector ``\psi_L`` is given by
```math
\psi_L \coloneqq \begin{pmatrix} -\sin(\alpha) \\ \cos(\alpha) \\ 0 \end{pmatrix}
```
where ``\alpha`` is the angle of attack.
This employs the convention that the wing is oriented such that the streamwise flow is in 
x-direction, the angle of attack rotates the flow into the y-direction, and that wing extends spanwise in the z-direction.

Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the the to-be-computed variables (for instance `LiftCoefficientPressure3D`) 
and boundary information.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rho_inf::Real`: Free-stream density
- `u_inf::Real`: Free-stream velocity
- `a_inf::Real`: Reference area of geometry (e.g. projected wing surface)
"""
function LiftCoefficientPressure3D(aoa, rho_inf, u_inf, a_inf)
    # `psi_lift` is the normal unit vector to the freestream direction.
    # Note: The choice of the normal vector `psi_lift = (-sin(aoa), cos(aoa), 0)`
    # leads to positive lift coefficients for positive angles of attack for airfoils.
    # One could also use `psi_lift = (sin(aoa), -cos(aoa), 0)` which results in the same
    # value, but with the opposite sign.
    psi_lift = (-sin(aoa), cos(aoa), zero(aoa))
    return LiftCoefficientPressure(ForceState(psi_lift, rho_inf, u_inf, a_inf))
end

@doc raw"""
    DragCoefficientPressure3D(aoa, rho_inf, u_inf, a_inf)

Compute the drag coefficient
```math
C_{D,p} \coloneqq \frac{\oint_{\partial \Omega} p \boldsymbol n \cdot \psi_D \, \mathrm{d} S}
                        {0.5 \rho_{\infty} U_{\infty}^2 A_{\infty}}
```
based on the pressure distribution along a boundary.
In 3D, the freestream-tangent unit vector ``\psi_D`` is given by
```math
\psi_D \coloneqq \begin{pmatrix} \cos(\alpha) \\ \sin(\alpha) \\ 0 \end{pmatrix}
```
where ``\alpha`` is the angle of attack.
This employs the convention that the wing is oriented such that the streamwise flow is in 
x-direction, the angle of attack rotates the flow into the y-direction, and that wing extends spanwise in the z-direction.

Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the the to-be-computed variables (for instance `DragCoefficientPressure3D`) 
and boundary information.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rho_inf::Real`: Free-stream density
- `u_inf::Real`: Free-stream velocity
- `a_inf::Real`: Reference area of geometry (e.g. projected wing surface)
"""
function DragCoefficientPressure3D(aoa, rho_inf, u_inf, a_inf)
    # `psi_drag` is the unit vector tangent to the freestream direction
    psi_drag = (cos(aoa), sin(aoa), zero(aoa))
    return DragCoefficientPressure(ForceState(psi_drag, rho_inf, u_inf, a_inf))
end

# 3D version of the `analyze` function for `AnalysisSurfaceIntegral`, i.e., 
# `LiftCoefficientPressure` and `DragCoefficientPressure`.
function analyze(surface_variable::AnalysisSurfaceIntegral, du, u, t,
                 mesh::P4estMesh{3},
                 equations, dg::DGSEM, cache, semi)
    @unpack boundaries = cache
    @unpack node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack variable, boundary_symbols = surface_variable
    @unpack boundary_symbol_indices = semi.boundary_conditions
    boundary_indices = get_boundary_indices(boundary_symbols, boundary_symbol_indices)

    # Restore boundary values for parabolic equations
    # which overwrite the solution boundary values with the gradients
    if semi isa SemidiscretizationHyperbolicParabolic
        prolong2boundaries!(cache, u, mesh, equations, dg)
    end

    surface_integral = zero(eltype(u))
    index_range = eachnode(dg)
    for boundary in boundary_indices
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_3d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_3d(node_indices[2], index_range)
        k_node_start, k_node_step = index_to_start_step_3d(node_indices[3], index_range)

        # In 3D, boundaries are surfaces => `node_index1`, `node_index2`
        for node_index1 in index_range
            # Reset node indices
            i_node = i_node_start
            j_node = j_node_start
            k_node = k_node_start
            for node_index2 in index_range
                u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg,
                                             node_index1, node_index2, boundary)
                # Extract normal direction at nodes which points from the elements outwards,
                # i.e., *into* the structure.
                normal_direction = get_normal_direction(direction,
                                                        contravariant_vectors,
                                                        i_node, j_node, k_node, element)

                # Coordinates at a boundary node
                x = get_node_coords(node_coordinates, equations, dg,
                                    i_node, j_node, k_node, element)

                # L2 norm of normal direction (contravariant_vector) is the surface element
                dS = weights[node_index1] * weights[node_index2] *
                     norm(normal_direction)

                # Integral over entire boundary surface. Note, it is assumed that the
                # `normal_direction` is normalized to be a normal vector within the
                # function `variable` and the division of the normal scaling factor
                # `norm(normal_direction)` is then accounted for with the `dS` quantity.
                surface_integral += variable(u_node, normal_direction, x, t,
                                             equations) * dS

                i_node += i_node_step
                j_node += j_node_step
                k_node += k_node_step
            end
        end
    end
    return surface_integral
end
end # muladd
