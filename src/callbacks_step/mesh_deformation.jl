# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    MeshDeformationCallback(semi; interval)

Performs mesh deformation every `interval` time steps
for a given semidiscretization `semi`.
"""
struct MeshDeformationCallback
    brep_folder::String
    interval::Int
end
# TODO additional members? caches?

function MeshDeformationCallback(; brep_folder, interval = 0)
    # check arguments
    if !(interval isa Integer && interval >= 0)
        throw(ArgumentError("`interval` must be a non-negative integer (provided `interval = $interval`)"))
    end

    md_callback = MeshDeformationCallback(brep_folder, interval)

    return DiscreteCallback(md_callback, md_callback, # the first one is the condition, the second the affect!
                            save_positions = (false, false),
                            initialize = initialize!)
end

function Base.show(io::IO, mime::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:MeshDeformationCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        amr_callback = cb.affect!

        summary_header(io, "MeshDeformationCallback")
        summary_line(io, "interval", amr_callback.interval)
        summary_footer(io)
    end
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: MeshDeformationCallback}
    md_callback = cb.affect!
    semi = integrator.p
    # TODO something to initialize?
    return nothing
end

# this method is called to determine whether the callback should be activated
function (md_callback::MeshDeformationCallback)(u, t, integrator)
    @unpack interval = md_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval > 0 && integrator.stats.naccept % interval == 0
end


# this method is called when the callback is activated
function (md_callback::MeshDeformationCallback)(integrator)
    @unpack brep_folder = md_callback

    u_ode = integrator.u
    semi = integrator.p

    @trixi_timeit timer() "Mesh deformation" begin
        
        @info "Calling t8_mesh_deformation..."
        Trixi.t8_mesh_deformation(semi.mesh.forest, brep_folder)
        
        adapt_to_deformation!(u_ode, mesh_equations_solver_cache(semi)...)
    end

    # avoid re-evaluating possible FSAL stages
    derivative_discontinuity!(integrator, false)
    return nothing
end

function adapt_to_deformation!(u_ode::AbstractVector, mesh::T8codeMesh,
                               equations, dg::DG, cache)
    @trixi_timeit timer() "reinitialize tree node coordinates" begin
        # Recalculate node coordinates of reference mesh.
        reinitialize_tree_node_coordinates!(mesh)
    end

    # Retain current solution and inverse Jacobian data.
    old_u_ode = copy(u_ode)
    old_inverse_jacobian = copy(cache.elements.inverse_jacobian)

    # OBS! If we don't GC.@preserve old_u_ode and old_inverse_jacobian, they might be GC'ed
    GC.@preserve old_u_ode begin
        old_u = wrap_array(old_u_ode, mesh, equations, dg, cache)
        u = wrap_array(u_ode, mesh, equations, dg, cache)

        # Reinitialize the mesh data structures
        @trixi_timeit timer() "reinitialize data structures" begin
            reinitialize_containers!(mesh, equations, dg, cache)
        end

        # Unlike AMR the numer of elements stays the same,
        # but we have to account for changes in volume
        @trixi_timeit timer() "rescale solution data" begin
            for element_id in 1:nelements(dg, cache)
                # Copy old element data to new element container, remove old Jacobian scaling,
                # apply new
                for v in eachvariable(equations)
                    u[v, .., element_id] .= (old_u[v, .., element_id] ./
                                            old_inverse_jacobian[.., element_id] .*
                                            cache.elements.inverse_jacobian[.., element_id])
                end
            end
        end
    end # GC.@preserve old_u_ode old_inverse_jacobian

    # TODO ?
    #reinitialize_boundaries!(semi.boundary_conditions, cache)

    mesh.unsaved_changes = true

    return nothing
end

function reinitialize_tree_node_coordinates!(mesh::T8codeMesh{2})
    @unpack forest, nodes, tree_node_coordinates = mesh

    # In t8code reference space is [0,1].
    nodes = 0.5f0 .* (nodes .+ 1)

    cmesh = t8_forest_get_cmesh(forest)
    number_of_trees = t8_forest_get_num_global_trees(forest)
    reference_coordinates = Vector{eltype(tree_node_coordinates)}(undef, 3)
    global_coordinates = Vector{eltype(tree_node_coordinates)}(undef, 3)

    for itree in 1:number_of_trees
        for j in eachindex(nodes), i in eachindex(nodes)
            reference_coordinates[1] = nodes[i]
            reference_coordinates[2] = nodes[j]
            reference_coordinates[3] = 0.0
            t8_geometry_evaluate(cmesh, itree - 1, reference_coordinates, 1,
                                 global_coordinates)
            @view(tree_node_coordinates[:, i, j, itree]) .= @view(global_coordinates[1:2])
        end
    end
end
end # @muladd
