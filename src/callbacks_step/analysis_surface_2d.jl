# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent
# TODO - Add explanation for what this file contains

struct AnalysisSurfaceIntegral{Semidiscretization, Indices, Variable}
    semi::Semidiscretization
    indices::Indices
    variable::Variable # Quantity of interest, like lift or drag
    function AnalysisSurfaceIntegral(semi, boundary_condition_type, variable)
        # The bc list as ordered in digest_boundary_conditions
        ordered_bc = semi.boundary_conditions.boundary_condition_types

        # The set of all indices that gives the bc where the surface integral is to be computed
        index = sort(findall(x -> x == boundary_condition_type, ordered_bc))

        # Put the bc in function form as they might change under AMR
        indices = semi -> Vector([semi.boundary_conditions.boundary_indices[i]
                                  for i in index])[1] # TODO - Should not need Vector and the [1]

        return new{typeof(semi), typeof(indices), typeof(variable)}(semi, indices,
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
    # psi_lift is the normal unit vector to the freestream direction
    psi_lift = (-sin(aoa), cos(aoa))
    force_state = ForceState(psi_lift, rhoinf, uinf, linf)
    return LiftCoefficient(force_state)
end

function DragCoefficient(aoa, rhoinf, uinf, linf)
    # psi_drag is the unit vector parallel to the freestream direction
    psi_drag = (cos(aoa), sin(aoa))
    return DragCoefficient(ForceState(psi_drag, rhoinf, uinf, linf))
end

function (lift_coefficient::LiftCoefficient)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack psi, rhoinf, uinf, linf = lift_coefficient.force_state
    n = dot(normal_direction, psi) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * linf)
end

function (drag_coefficient::DragCoefficient)(u, normal_direction, equations)
    p = pressure(u, equations)
    @unpack psi, rhoinf, uinf, linf = drag_coefficient.force_state
    n = dot(normal_direction, psi) / norm(normal_direction)
    return p * n / (0.5 * rhoinf * uinf^2 * linf)
end

function analyze(surface_variable::AnalysisSurfaceIntegral, du, u, t,
                 mesh::P4estMesh{2},
                 equations, dg::DGSEM, cache)
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis
    @unpack semi, indices, variable = surface_variable
    indices_ = indices(semi)

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
            dS = weights[node_index] * norm(normal_direction)
            surface_integral += variable(u_node, normal_direction, equations) * dS

            i_node += i_node_step
            j_node += j_node_step
        end
    end
    return surface_integral
end

function pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, <:Any,
                                                     <:LiftCoefficient{<:Any}})
    "CL"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, <:Any,
                                                   <:LiftCoefficient{<:Any}})
    "CL"
end
function pretty_form_ascii(::AnalysisSurfaceIntegral{<:Any, <:Any,
                                                     <:DragCoefficient{<:Any}})
    "CD"
end
function pretty_form_utf(::AnalysisSurfaceIntegral{<:Any, <:Any,
                                                   <:DragCoefficient{<:Any}})
    "CD"
end
end # muladd
