
struct AnalysisSurfaceIntegral{Indices, Variable}
    indices::Indices
    variable::Variable
end

struct ForceState{RealT <: Real}
    Ψ::Tuple{RealT, RealT} # Unit vector normal or parallel to freestream
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

struct LiftCoefficient{RealT <: Real}
    force_state::ForceState{RealT}
end

struct DragCoefficient{RealT <: Real}
    force_state::ForceState{RealT}
end

function LiftCoefficient(aoa, rhoinf, uinf, linf)
    # Ψl is the normal unit vector to the freestream direction
    Ψl = (-sin(aoa), cos(aoa))
    force_state = ForceState(Ψl, rhoinf, uinf, linf)
    return LiftCoefficient(force_state)
end

function DragCoefficient(aoa, rhoinf, uinf, linf)
    # Ψd is the unit vector parallel to the freestream direction
    Ψd = (cos(aoa), sin(aoa))
    return DragCoefficient(ForceState(Ψd, rhoinf, uinf, linf))
end

function (lift_coefficient::LiftCoefficient)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack Ψ, rhoinf, uinf, linf = lift_coefficient.force_state
    n = dot(normal_direction, Ψ) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * linf)
end

function (drag_coefficient::DragCoefficient)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack Ψ, rhoinf, uinf, linf = drag_coefficient.force_state
    n = dot(normal_direction, Ψ) / norm(normal_direction)
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

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, <:LiftCoefficient{<:Any}})
    "CL"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, <:LiftCoefficient{<:Any}})
    "CL"
end
function pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, <:DragCoefficient{<:Any}})
    "CD"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, <:DragCoefficient{<:Any}})
    "CD"
end
