# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file contains callbacks that are performed on the surface like computation of
# surface forces

"""
    AnalysisSurfaceIntegral{Semidiscretization, Variable}(semi,
                                                          boundary_symbol_or_boundary_symbols,
                                                          variable)

    This struct is used to compute the surface integral of a quantity of interest `variable` alongside
    the boundary/boundaries associated with particular name(s) given in `boundary_symbol`
    or `boundary_symbols`.
    For instance, this can be used to compute the lift [`LiftCoefficientPressure`](@ref) or
    drag coefficient [`DragCoefficientPressure`](@ref) of e.g. an airfoil with the boundary
    name `:Airfoil` in 2D.
"""
struct AnalysisSurfaceIntegral{Semidiscretization, Variable}
    semi::Semidiscretization # passed in to retrieve boundary condition information
    indices::Vector{Int} # Indices in `boundary_condition_indices` where quantity of interest is computed
    variable::Variable # Quantity of interest, like lift or drag

    function AnalysisSurfaceIntegral(semi, boundary_symbol, variable)
        @unpack boundary_symbol_indices = semi.boundary_conditions
        indices = boundary_symbol_indices[boundary_symbol]

        return new{typeof(semi), typeof(variable)}(semi, indices,
                                                   variable)
    end

    function AnalysisSurfaceIntegral(semi, boundary_symbols::Vector{Symbol}, variable)
        @unpack boundary_symbol_indices = semi.boundary_conditions
        indices = Vector{Int}()
        for name in boundary_symbols
            append!(indices, boundary_symbol_indices[name])
        end
        sort!(indices)

        return new{typeof(semi), typeof(variable)}(semi, indices,
                                                   variable)
    end
end

struct ForceState{RealT <: Real}
    psi::Tuple{RealT, RealT} # Unit vector normal or parallel to freestream
    rhoinf::RealT
    uinf::RealT
    l_inf::RealT
end

struct LiftCoefficientPressure{RealT <: Real}
    force_state::ForceState{RealT}
end

struct DragCoefficientPressure{RealT <: Real}
    force_state::ForceState{RealT}
end

"""
    LiftCoefficientPressure(aoa, rhoinf, uinf, l_inf)

Compute the lift coefficient
```math
C_{L,p} \\coloneqq \\frac{\\oint_{\\partial \\Omega} p \\boldsymbol n \\cdot \\psi_L \\, \\mathrm{d} S}
                        {0.5 \\cdot \\rho_{\\infty} \\cdot U_{\\infty}^2 \\cdot L_{\\infty}}
```
based on the pressure distribution along a boundary.
Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the boundary information and semidiscretization.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rhoinf::Real`: Free-stream density
- `uinf::Real`: Free-stream velocity
- `l_inf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function LiftCoefficientPressure(aoa, rhoinf, uinf, l_inf)
    # psi_lift is the normal unit vector to the freestream direction.
    # Note: The choice of the normal vector psi_lift = (-sin(aoa), cos(aoa))
    # leads to positive lift coefficients for positive angles of attack for airfoils.
    # One could also use psi_lift = (sin(aoa), -cos(aoa)) which results in the same
    # value, but with the opposite sign.
    psi_lift = (-sin(aoa), cos(aoa))
    return LiftCoefficientPressure(ForceState(psi_lift, rhoinf, uinf, l_inf))
end

"""
    DragCoefficientPressure(aoa, rhoinf, uinf, l_inf)

Compute the drag coefficient
```math
C_{D,p} \\coloneqq \\frac{\\oint_{\\partial \\Omega} p \\boldsymbol n \\cdot \\psi_D \\, \\mathrm{d} S}
                        {0.5 \\cdot \\rho_{\\infty} \\cdot U_{\\infty}^2 \\cdot L_{\\infty}}
```
based on the pressure distribution along a boundary.
Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the boundary information and semidiscretization.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rhoinf::Real`: Free-stream density
- `uinf::Real`: Free-stream velocity
- `l_inf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function DragCoefficientPressure(aoa, rhoinf, uinf, l_inf)
    # `psi_drag` is the unit vector tangent to the freestream direction
    psi_drag = (cos(aoa), sin(aoa))
    return DragCoefficientPressure(ForceState(psi_drag, rhoinf, uinf, l_inf))
end

function (lift_coefficient::LiftCoefficientPressure)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack psi, rhoinf, uinf, l_inf = lift_coefficient.force_state
    # Normalize as `normal_direction` is not necessarily a unit vector
    n = dot(normal_direction, psi) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * l_inf)
end

function (drag_coefficient::DragCoefficientPressure)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack psi, rhoinf, uinf, l_inf = drag_coefficient.force_state
    # Normalize as `normal_direction` is not necessarily a unit vector
    n = dot(normal_direction, psi) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * l_inf)
end

function analyze(surface_variable::AnalysisSurfaceIntegral, du, u, t,
                 mesh::P4estMesh{2},
                 equations, dg::DGSEM, cache)
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack indices, variable = surface_variable

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

            # L2 norm of normal direction (contravariant_vector) is the surface element
            dS = weights[node_index] * norm(normal_direction)
            # Integral over entire boundary surface
            surface_integral += variable(u_node, normal_direction, equations) * dS

            i_node += i_node_step
            j_node += j_node_step
        end
    end
    return surface_integral
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any,
                                                     <:LiftCoefficientPressure{<:Any}})
    "CL_p"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:Any,
                                                   <:LiftCoefficientPressure{<:Any}})
    "CL_p"
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any,
                                                     <:DragCoefficientPressure{<:Any}})
    "CD_p"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:Any,
                                                   <:DragCoefficientPressure{<:Any}})
    "CD_p"
end
end # muladd
