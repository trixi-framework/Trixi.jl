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
    interval::Int
end
# TODO additional members? caches?

function MeshDeformationCallback(; interval = 0)
    # check arguments
    if !(interval isa Integer && interval >= 0)
        throw(ArgumentError("`interval` must be a non-negative integer (provided `interval = $interval`)"))
    end

    md_callback = MeshDeformationCallback(interval)

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
function (amr_callback::MeshDeformationCallback)(integrator)
    u_ode = integrator.u
    semi = integrator.p

    @trixi_timeit timer() "Mesh deformation" begin
        
        @info "Do mesh deformation here!"
        
        adapt_to_deformation!(u_ode, mesh_equations_solver_cache(semi)...)
    end

    # avoid re-evaluating possible FSAL stages
    derivative_discontinuity!(integrator, false)
    return nothing
end

function adapt_to_deformation!(u_ode::AbstractVector, mesh::T8codeMesh,
                               equations, dg::DG, cache)
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
end # @muladd
