# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file contains callbacks that are performed on the surface like computation of
# surface forces

"""
    AnalysisSurfaceIntegral{Variable, NBoundaries}(semi,
                                                   boundary_symbols::NTuple{NBoundaries, Symbol},
                                                   variable)

This struct is used to compute the surface integral of a quantity of interest `variable` alongside
the boundary/boundaries associated with particular name(s) given in `boundary_symbol`
or `boundary_symbols`.
For instance, this can be used to compute the lift [`LiftCoefficientPressure`](@ref) or
drag coefficient [`DragCoefficientPressure`](@ref) of e.g. an airfoil with the boundary
name `:Airfoil` in 2D.

- `boundary_symbols::NTuple{NBoundaries, Symbol}`: Name(s) of the boundary/boundaries
  where the quantity of interest is computed
- `variable::Variable`: Quantity of interest, like lift or drag
"""
struct AnalysisSurfaceIntegral{Variable, NBoundaries}
    variable::Variable # Quantity of interest, like lift or drag
    boundary_symbols::NTuple{NBoundaries, Symbol} # Name(s) of the boundary/boundaries

    function AnalysisSurfaceIntegral(boundary_symbols::NTuple{NBoundaries, Symbol},
                                     variable) where {NBoundaries}
        return new{typeof(variable), NBoundaries}(variable, boundary_symbols)
    end
end

struct ForceState{RealT <: Real}
    psi::Tuple{RealT, RealT} # Unit vector normal or parallel to freestream
    rhoinf::RealT
    uinf::RealT
    linf::RealT
end

struct LiftCoefficientPressure{RealT <: Real}
    force_state::ForceState{RealT}
end

struct DragCoefficientPressure{RealT <: Real}
    force_state::ForceState{RealT}
end

# Abstract base type used for dispatch of `analyze` for quantities
# requiring gradients of the velocity field.
abstract type VariableViscous end

struct LiftCoefficientShearStress{RealT <: Real} <: VariableViscous
    force_state::ForceState{RealT}
end

struct DragCoefficientShearStress{RealT <: Real} <: VariableViscous
    force_state::ForceState{RealT}
end

"""
    LiftCoefficientPressure(aoa, rhoinf, uinf, linf)

Compute the lift coefficient
```math
C_{L,p} \\coloneqq \\frac{\\oint_{\\partial \\Omega} p \\boldsymbol n \\cdot \\psi_L \\, \\mathrm{d} S}
                        {0.5 \\rho_{\\infty} U_{\\infty}^2 L_{\\infty}}
```
based on the pressure distribution along a boundary.
Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the boundary information and semidiscretization.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rhoinf::Real`: Free-stream density
- `uinf::Real`: Free-stream velocity
- `linf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function LiftCoefficientPressure(aoa, rhoinf, uinf, linf)
    # psi_lift is the normal unit vector to the freestream direction.
    # Note: The choice of the normal vector psi_lift = (-sin(aoa), cos(aoa))
    # leads to positive lift coefficients for positive angles of attack for airfoils.
    # One could also use psi_lift = (sin(aoa), -cos(aoa)) which results in the same
    # value, but with the opposite sign.
    psi_lift = (-sin(aoa), cos(aoa))
    return LiftCoefficientPressure(ForceState(psi_lift, rhoinf, uinf, linf))
end

"""
    DragCoefficientPressure(aoa, rhoinf, uinf, linf)

Compute the drag coefficient
```math
C_{D,p} \\coloneqq \\frac{\\oint_{\\partial \\Omega} p \\boldsymbol n \\cdot \\psi_D \\, \\mathrm{d} S}
                        {0.5 \\rho_{\\infty} U_{\\infty}^2 L_{\\infty}}
```
based on the pressure distribution along a boundary.
Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the boundary information and semidiscretization.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rhoinf::Real`: Free-stream density
- `uinf::Real`: Free-stream velocity
- `linf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function DragCoefficientPressure(aoa, rhoinf, uinf, linf)
    # `psi_drag` is the unit vector tangent to the freestream direction
    psi_drag = (cos(aoa), sin(aoa))
    return DragCoefficientPressure(ForceState(psi_drag, rhoinf, uinf, linf))
end

"""
    LiftCoefficientShearStress(aoa, rhoinf, uinf, linf)

Compute the lift coefficient
```math
C_{L,f} \\coloneqq \\frac{\\oint_{\\partial \\Omega} \\boldsymbol \\tau_w \\cdot \\psi_L \\, \\mathrm{d} S}
                        {0.5 \\rho_{\\infty} U_{\\infty}^2 L_{\\infty}}
```
based on the wall shear stress vector ``\\tau_w`` along a boundary.
Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the boundary information and semidiscretization.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rhoinf::Real`: Free-stream density
- `uinf::Real`: Free-stream velocity
- `linf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function LiftCoefficientShearStress(aoa, rhoinf, uinf, linf)
    # psi_lift is the normal unit vector to the freestream direction.
    # Note: The choice of the normal vector psi_lift = (-sin(aoa), cos(aoa))
    # leads to negative lift coefficients for airfoils.
    # One could also use psi_lift = (sin(aoa), -cos(aoa)) which results in the same
    # value, but with the opposite sign.
    psi_lift = (-sin(aoa), cos(aoa))
    return LiftCoefficientShearStress(ForceState(psi_lift, rhoinf, uinf, linf))
end

"""
    DragCoefficientShearStress(aoa, rhoinf, uinf, linf)

Compute the drag coefficient
```math
C_{D,f} \\coloneqq \\frac{\\oint_{\\partial \\Omega} \\boldsymbol \\tau_w \\cdot \\psi_D \\, \\mathrm{d} S}
                        {0.5 \\rho_{\\infty} U_{\\infty}^2 L_{\\infty}}
```
based on the wall shear stress vector ``\\tau_w`` along a boundary.
Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the boundary information and semidiscretization.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rhoinf::Real`: Free-stream density
- `uinf::Real`: Free-stream velocity
- `linf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function DragCoefficientShearStress(aoa, rhoinf, uinf, linf)
    # `psi_drag` is the unit vector tangent to the freestream direction
    psi_drag = (cos(aoa), sin(aoa))
    return DragCoefficientShearStress(ForceState(psi_drag, rhoinf, uinf, linf))
end

function (lift_coefficient::LiftCoefficientPressure)(u, normal_direction, x, t,
                                                     equations)
    p = pressure(u, equations)
    @unpack psi, rhoinf, uinf, linf = lift_coefficient.force_state
    # Normalize as `normal_direction` is not necessarily a unit vector
    n = dot(normal_direction, psi) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * linf)
end

function (drag_coefficient::DragCoefficientPressure)(u, normal_direction, x, t,
                                                     equations)
    p = pressure(u, equations)
    @unpack psi, rhoinf, uinf, linf = drag_coefficient.force_state
    # Normalize as `normal_direction` is not necessarily a unit vector
    n = dot(normal_direction, psi) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * linf)
end

# Compute the three components of the symmetric viscous stress tensor
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

function viscous_stress_vector(u, normal_direction, equations_parabolic,
                               gradients_1, gradients_2)
    #  Normalize normal direction, should point *into* the fluid => *(-1)
    n_normal = -normal_direction / norm(normal_direction)

    tau_11, tau_12, tau_22 = viscous_stress_tensor(u, normal_direction,
                                                   equations_parabolic,
                                                   gradients_1, gradients_2)

    # Viscous stress vector: Stress tensor * normal vector
    visc_stress_vector_1 = tau_11 * n_normal[1] + tau_12 * n_normal[2]
    visc_stress_vector_2 = tau_12 * n_normal[1] + tau_22 * n_normal[2]

    return (visc_stress_vector_1, visc_stress_vector_2)
end

function (lift_coefficient::LiftCoefficientShearStress)(u, normal_direction, x, t,
                                                        equations_parabolic,
                                                        gradients_1, gradients_2)
    visc_stress_vector = viscous_stress_vector(u, normal_direction, equations_parabolic,
                                               gradients_1, gradients_2)
    @unpack psi, rhoinf, uinf, linf = lift_coefficient.force_state
    return (visc_stress_vector[1] * psi[1] + visc_stress_vector[2] * psi[2]) /
           (0.5 * rhoinf * uinf^2 * linf)
end

function (drag_coefficient::DragCoefficientShearStress)(u, normal_direction, x, t,
                                                        equations_parabolic,
                                                        gradients_1, gradients_2)
    visc_stress_vector = viscous_stress_vector(u, normal_direction, equations_parabolic,
                                               gradients_1, gradients_2)
    @unpack psi, rhoinf, uinf, linf = drag_coefficient.force_state
    return (visc_stress_vector[1] * psi[1] + visc_stress_vector[2] * psi[2]) /
           (0.5 * rhoinf * uinf^2 * linf)
end

function get_boundary_indices(boundary_symbols, boundary_symbol_indices)
    indices = Int[]
    for name in boundary_symbols
        append!(indices, boundary_symbol_indices[name])
    end
    sort!(indices) # Try to achieve some data locality by sorting

    return indices
end

function analyze(surface_variable::AnalysisSurfaceIntegral, du, u, t,
                 mesh::P4estMesh{2},
                 equations, dg::DGSEM, cache, semi)
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack variable, boundary_symbols = surface_variable
    @unpack boundary_symbol_indices = semi.boundary_conditions
    indices = get_boundary_indices(boundary_symbols, boundary_symbol_indices)

    surface_integral = zero(eltype(u))
    index_range = eachnode(dg)
    for boundary in indices
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in index_range
            u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg, node_index,
                                         boundary)
            # Extract normal direction at nodes which points from the elements outwards,
            # i.e., *into* the structure.
            normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node,
                                                    element)

            # Coordinates at a boundary node
            x = get_node_coords(node_coordinates, equations, dg, i_node, j_node,
                                element)

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

function analyze(surface_variable::AnalysisSurfaceIntegral{Variable},
                 du, u, t, mesh::P4estMesh{2},
                 equations, equations_parabolic,
                 dg::DGSEM, cache, semi,
                 cache_parabolic) where {Variable <: VariableViscous}
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack variable, boundary_symbols = surface_variable
    @unpack boundary_symbol_indices = semi.boundary_conditions
    indices = get_boundary_indices(boundary_symbols, boundary_symbol_indices)

    # Additions for parabolic
    @unpack viscous_container = cache_parabolic
    @unpack gradients = viscous_container

    gradients_x, gradients_y = gradients

    surface_integral = zero(eltype(u))
    index_range = eachnode(dg)
    for boundary in indices
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in index_range
            u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg, node_index,
                                         boundary)
            # Extract normal direction at nodes which points from the elements outwards,
            # i.e., *into* the structure.
            normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node,
                                                    element)

            # Coordinates at a boundary node
            x = get_node_coords(node_coordinates, equations, dg, i_node, j_node,
                                element)

            # L2 norm of normal direction (contravariant_vector) is the surface element
            dS = weights[node_index] * norm(normal_direction)

            gradients_1 = get_node_vars(gradients_x, equations_parabolic, dg, i_node,
                                        j_node, element)
            gradients_2 = get_node_vars(gradients_y, equations_parabolic, dg, i_node,
                                        j_node, element)

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

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:LiftCoefficientPressure{<:Any}})
    "CL_p"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:LiftCoefficientPressure{<:Any}})
    "CL_p"
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:DragCoefficientPressure{<:Any}})
    "CD_p"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:DragCoefficientPressure{<:Any}})
    "CD_p"
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:LiftCoefficientShearStress{<:Any}})
    "CL_f"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:LiftCoefficientShearStress{<:Any}})
    "CL_f"
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:DragCoefficientShearStress{<:Any}})
    "CD_f"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:DragCoefficientShearStress{<:Any}})
    "CD_f"
end
end # muladd
