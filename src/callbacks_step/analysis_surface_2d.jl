using Trixi
using Trixi: integrate_via_indices, norm, apply_jacobian_parabolic!, @threaded,
             indices2direction,
             index_to_start_step_2d, get_normal_direction, dot, get_node_coords
import Trixi: analyze, pretty_form_ascii, pretty_form_utf

struct AnalysisSurfaceIntegral{Indices, Variable}
    indices::Indices
    variable::Variable
end

struct ForceState{RealT <: Real}
    Ψl::Tuple{RealT, RealT}
    rhoinf::RealT
    uinf::RealT
    linf::RealT
end

# TODO - This should be a struct in ForceState
struct FreeStreamVariables{RealT <: Real}
    rhoinf::RealT
    uinf::RealT
    linf::RealT
end

struct LiftForcePressure{RealT <: Real}
    force_state::ForceState{RealT}
end

struct DragForcePressure{RealT <: Real}
    force_state::ForceState{RealT}
end

function LiftForcePressure(aoa::Real, rhoinf::Real, uinf::Real, linf::Real)
    Ψl = (-sin(aoa), cos(aoa))
    force_state = ForceState(Ψl, rhoinf, uinf, linf)
    return LiftForcePressure(force_state)
end

function DragForcePressure(aoa::Real, rhoinf::Real, uinf::Real, linf::Real)
    Ψd = (cos(aoa), sin(aoa))
    return DragForcePressure(ForceState(Ψd, rhoinf, uinf, linf))
end

function (lift_force::LiftForcePressure)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack Ψl, rhoinf, uinf, linf = lift_force.force_state
    n = dot(normal_direction, Ψl) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * linf)
end

function (drag_force::DragForcePressure)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack Ψl, rhoinf, uinf, linf = drag_force.force_state
    n = dot(normal_direction, Ψl) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * linf)
end

function analyze(surface_variable::AnalysisSurfaceIntegral, du, u, t,
                 mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                 equations, dg::DGSEM, cache)
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis
    @unpack indices, variable = surface_variable
    # TODO - Use initialize callbacks to move boundary_conditions to cache
    indices_ = indices(cache)

    surface_integral = zero(eltype(u))
    index_range = eachnode(dg)
    for local_index in eachindex(indices_)
        # Use the local index to get the global boundary index from the pre-sorted list
        boundary = indices_[local_index]

        # Get information on the adjacent element, compute the surface fluxes,
        # and store them
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in eachnode(dg)
            u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg, node_index,
                                         boundary)
            normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node,
                                                    element)

            # L2 norm of normal direction is the surface element
            # 0.5 factor is NOT needed, the norm(normal_direction) is all the factor needed
            dS = weights[node_index] * norm(normal_direction)
            surface_integral += variable(u_node, normal_direction, equations) * dS

            i_node += i_node_step
            j_node += j_node_step
        end
    end
    return surface_integral
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, <:LiftForcePressure{<:Any}})
    "Pressure_lift"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, <:LiftForcePressure{<:Any}})
    "Pressure_lift"
end
function pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, <:DragForcePressure{<:Any}})
    "Pressure_drag"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, <:DragForcePressure{<:Any}})
    "Pressure_drag"
end
