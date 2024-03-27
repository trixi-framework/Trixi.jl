# This file contains callbacks that are performed on the surface like computation of
# surface forces

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    AnalysisSurfaceIntegral{Semidiscretization, Variable}(semi, 
                                                          boundary_symbol_or_boundary_symbols, 
                                                          variable)

    This struct is used to compute the surface integral of a quantity of interest `variable` alongside 
    the boundary/boundaries associated with `boundary_symbol` or `boundary_symbols`.
    For instance, this can be used to compute the lift or drag coefficient of e.g. an airfoil in 2D.
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
    linf::RealT
end

struct LiftCoefficient{RealT <: Real}
    force_state::ForceState{RealT}
end

struct DragCoefficient{RealT <: Real}
    force_state::ForceState{RealT}
end

function LiftCoefficient(aoa, rhoinf, uinf, linf)
    # psi_lift is the normal unit vector to the freestream direction.
    # Note: The choice of the normal vector psi_lift = (-sin(aoa), cos(aoa))
    # leads to positive lift coefficients for positive angles of attack for airfoils.
    # Note that one could also use psi_lift = (sin(aoa), -cos(aoa)) which results in the same 
    # value, but with the opposite sign.
    psi_lift = (-sin(aoa), cos(aoa))
    return LiftCoefficient(ForceState(psi_lift, rhoinf, uinf, linf))
end

function DragCoefficient(aoa, rhoinf, uinf, linf)
    # `psi_drag` is the unit vector in direction of the freestream.
    psi_drag = (cos(aoa), sin(aoa))
    return DragCoefficient(ForceState(psi_drag, rhoinf, uinf, linf))
end

function (lift_coefficient::LiftCoefficient)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack psi, rhoinf, uinf, linf = lift_coefficient.force_state
    # Normalize as `normal_direction` is not necessarily a unit vector
    n = dot(normal_direction, psi) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * linf)
end

function (drag_coefficient::DragCoefficient)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack psi, rhoinf, uinf, linf = drag_coefficient.force_state
    # Normalize as `normal_direction` is not necessarily a unit vector
    n = dot(normal_direction, psi) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * linf)
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
            # Integral over whole boundary surface
            surface_integral += variable(u_node, normal_direction, equations) * dS

            i_node += i_node_step
            j_node += j_node_step
        end
    end
    return surface_integral
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any,
                                                     <:LiftCoefficient{<:Any}})
    "CL"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:Any,
                                                   <:LiftCoefficient{<:Any}})
    "CL"
end
function pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any,
                                                     <:DragCoefficient{<:Any}})
    "CD"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:Any,
                                                   <:DragCoefficient{<:Any}})
    "CD"
end
end # muladd
