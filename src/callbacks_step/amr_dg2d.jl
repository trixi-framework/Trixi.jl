# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Redistribute data for load balancing after partitioning the mesh
function rebalance_solver!(u_ode::AbstractVector, mesh::TreeMesh{2}, equations,
                           dg::DGSEM, cache, old_mpi_ranks_per_cell)
    if cache.elements.cell_ids == local_leaf_cells(mesh.tree)
        # Cell ids of the current elements are the same as the local leaf cells of the
        # newly partitioned mesh, so the solver doesn't need to be rebalanced on this rank.
        # MPICache init uses all-to-all communication -> reinitialize even if there is nothing to do
        # locally (there are other MPI ranks that need to be rebalanced if this function is called)
        reinitialize_containers!(mesh, equations, dg, cache)
        return
    end

    # Retain current solution data
    old_n_elements = nelements(dg, cache)
    old_cell_ids = copy(cache.elements.cell_ids)
    old_u_ode = copy(u_ode)
    GC.@preserve old_u_ode begin # OBS! If we don't GC.@preserve old_u_ode, it might be GC'ed
        # Use `wrap_array_native` instead of `wrap_array` since MPI might not interact
        # nicely with non-base array types
        old_u = wrap_array_native(old_u_ode, mesh, equations, dg, cache)

        @trixi_timeit timer() "reinitialize data structures" begin
            reinitialize_containers!(mesh, equations, dg, cache)
        end

        resize!(u_ode,
                nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache))
        u = wrap_array_native(u_ode, mesh, equations, dg, cache)

        # Get new list of leaf cells
        leaf_cell_ids = local_leaf_cells(mesh.tree)

        @trixi_timeit timer() "exchange data" begin
            # Collect MPI requests for MPI_Waitall
            requests = Vector{MPI.Request}()

            # Find elements that will change their rank and send their data to the new rank
            for old_element_id in 1:old_n_elements
                cell_id = old_cell_ids[old_element_id]
                if !(cell_id in leaf_cell_ids)
                    # Send element data to new rank, use cell_id as tag (non-blocking)
                    dest = mesh.tree.mpi_ranks[cell_id]
                    request = MPI.Isend(@view(old_u[:, .., old_element_id]), dest,
                                        cell_id, mpi_comm())
                    push!(requests, request)
                end
            end

            # Loop over all elements in new container and either copy them from old container
            # or receive them with MPI
            for element in eachelement(dg, cache)
                cell_id = cache.elements.cell_ids[element]
                if cell_id in old_cell_ids
                    old_element_id = searchsortedfirst(old_cell_ids, cell_id)
                    # Copy old element data to new element container
                    @views u[:, .., element] .= old_u[:, .., old_element_id]
                else
                    # Receive old element data
                    src = old_mpi_ranks_per_cell[cell_id]
                    request = MPI.Irecv!(@view(u[:, .., element]), src, cell_id,
                                         mpi_comm())
                    push!(requests, request)
                end
            end

            # Wait for all non-blocking MPI send/receive operations to finish
            MPI.Waitall(requests, MPI.Status)
        end
    end # GC.@preserve old_u_ode
end

# Refine elements in the DG solver based on a list of cell_ids that should be refined.
# On `P4estMesh`, if an element refines the solution scaled by the Jacobian `J*u` is interpolated
# from the parent element into the four children elements. The solution on each child
# element is then recovered by dividing by the new element Jacobians.
function refine!(u_ode::AbstractVector, adaptor, mesh::Union{TreeMesh{2}, P4estMesh{2}},
                 equations, dg::DGSEM, cache, elements_to_refine)
    # Return early if there is nothing to do
    if isempty(elements_to_refine)
        if mpi_isparallel()
            # MPICache init uses all-to-all communication -> reinitialize even if there is nothing to do
            # locally (there still might be other MPI ranks that have refined elements)
            reinitialize_containers!(mesh, equations, dg, cache)
        end
        return
    end

    # Determine for each existing element whether it needs to be refined
    needs_refinement = falses(nelements(dg, cache))
    needs_refinement[elements_to_refine] .= true

    # Retain current solution data
    old_n_elements = nelements(dg, cache)
    old_u_ode = copy(u_ode)
    old_inverse_jacobian = copy(cache.elements.inverse_jacobian)
    # OBS! If we don't GC.@preserve old_u_ode and old_inverse_jacobian, they might be GC'ed
    GC.@preserve old_u_ode old_inverse_jacobian begin
        old_u = wrap_array(old_u_ode, mesh, equations, dg, cache)

        if mesh isa P4estMesh
            # Loop over all elements in old container and scale the old solution by the Jacobian
            # prior to projection
            for old_element_id in 1:old_n_elements
                for v in eachvariable(equations)
                    old_u[v, .., old_element_id] .= (old_u[v, .., old_element_id] ./
                                                     old_inverse_jacobian[..,
                                                                          old_element_id])
                end
            end
        end

        reinitialize_containers!(mesh, equations, dg, cache)

        resize!(u_ode,
                nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache))
        u = wrap_array(u_ode, mesh, equations, dg, cache)

        # Loop over all elements in old container and either copy them or refine them
        element_id = 1
        for old_element_id in 1:old_n_elements
            if needs_refinement[old_element_id]
                # Refine element and store solution directly in new data structure
                refine_element!(u, element_id, old_u, old_element_id,
                                adaptor, equations, dg)

                if mesh isa P4estMesh
                    # Before `element_id` is incremented, divide by the new Jacobians on each
                    # child element and save the result
                    for m in 0:3 # loop over the children
                        for v in eachvariable(equations)
                            u[v, .., element_id + m] .*= (0.25f0 .*
                                                          cache.elements.inverse_jacobian[..,
                                                                                          element_id + m])
                        end
                    end
                end

                # Increment `element_id` on the refined mesh with the number
                # of children, i.e., 4 in 2D
                element_id += 2^ndims(mesh)
            else
                if mesh isa P4estMesh
                    # Copy old element data to new element container and remove Jacobian scaling
                    for v in eachvariable(equations)
                        u[v, .., element_id] .= (old_u[v, .., old_element_id] .*
                                                 old_inverse_jacobian[..,
                                                                      old_element_id])
                    end
                else # isa TreeMesh
                    @views u[:, .., element_id] .= old_u[:, .., old_element_id]
                end
                # No refinement occurred, so increment `element_id` on the new mesh by one
                element_id += 1
            end
        end
        # If everything is correct, we should have processed all elements.
        # Depending on whether the last element processed above had to be refined or not,
        # the counter `element_id` can have two different values at the end.
        @assert element_id ==
                nelements(dg, cache) +
                1||element_id == nelements(dg, cache) + 2^ndims(mesh) "element_id = $element_id, nelements(dg, cache) = $(nelements(dg, cache))"
    end # GC.@preserve old_u_ode old_inverse_jacobian

    # Sanity check
    if mesh isa TreeMesh && isperiodic(mesh.tree) && nmortars(cache.mortars) == 0 &&
       !mpi_isparallel()
        @assert ninterfaces(cache.interfaces)==ndims(mesh) * nelements(dg, cache) ("For $(ndims(mesh))D and periodic domains and conforming elements, the number of interfaces must be $(ndims(mesh)) times the number of elements")
    end

    return nothing
end

function refine!(u_ode::AbstractVector, adaptor,
                 mesh::Union{TreeMesh{2}, P4estMesh{2}, TreeMesh{3}, P4estMesh{3}},
                 equations, dg::DGSEM, cache, cache_parabolic,
                 elements_to_refine)
    # Call `refine!` for the hyperbolic part, which does the heavy lifting of
    # actually transferring the solution to the refined cells
    refine!(u_ode, adaptor, mesh, equations, dg, cache, elements_to_refine)

    # Resize parabolic helper variables
    @unpack viscous_container = cache_parabolic
    resize!(viscous_container, equations, dg, cache)
    reinitialize_containers!(mesh, equations, dg, cache_parabolic)

    # Sanity check
    if mesh isa TreeMesh && isperiodic(mesh.tree) && nmortars(cache.mortars) == 0 &&
       !mpi_isparallel()
        @assert ninterfaces(cache_parabolic.interfaces)==ndims(mesh) *
                                                         nelements(dg, cache_parabolic) ("For $(ndims(mesh))D and periodic domains and conforming elements, the number of interfaces must be $(ndims(mesh)) times the number of elements")
    end

    return nothing
end

# TODO: Taal compare performance of different implementations
# Refine solution data u for an element, using L2 projection (interpolation)
function refine_element!(u::AbstractArray{<:Any, 4}, element_id,
                         old_u, old_element_id,
                         adaptor::LobattoLegendreAdaptorL2, equations, dg)
    @unpack forward_upper, forward_lower = adaptor

    # Store new element ids
    lower_left_id = element_id
    lower_right_id = element_id + 1
    upper_left_id = element_id + 2
    upper_right_id = element_id + 3

    @boundscheck begin
        @assert old_element_id >= 1
        @assert size(old_u, 1) == nvariables(equations)
        @assert size(old_u, 2) == nnodes(dg)
        @assert size(old_u, 3) == nnodes(dg)
        @assert size(old_u, 4) >= old_element_id
        @assert element_id >= 1
        @assert size(u, 1) == nvariables(equations)
        @assert size(u, 2) == nnodes(dg)
        @assert size(u, 3) == nnodes(dg)
        @assert size(u, 4) >= element_id + 3
    end

    # Interpolate to lower left element
    for j in eachnode(dg), i in eachnode(dg)
        acc = zero(get_node_vars(u, equations, dg, i, j, element_id))
        for l in eachnode(dg), k in eachnode(dg)
            acc += get_node_vars(old_u, equations, dg, k, l, old_element_id) *
                   forward_lower[i, k] * forward_lower[j, l]
        end
        set_node_vars!(u, acc, equations, dg, i, j, lower_left_id)
    end

    # Interpolate to lower right element
    for j in eachnode(dg), i in eachnode(dg)
        acc = zero(get_node_vars(u, equations, dg, i, j, element_id))
        for l in eachnode(dg), k in eachnode(dg)
            acc += get_node_vars(old_u, equations, dg, k, l, old_element_id) *
                   forward_upper[i, k] * forward_lower[j, l]
        end
        set_node_vars!(u, acc, equations, dg, i, j, lower_right_id)
    end

    # Interpolate to upper left element
    for j in eachnode(dg), i in eachnode(dg)
        acc = zero(get_node_vars(u, equations, dg, i, j, element_id))
        for l in eachnode(dg), k in eachnode(dg)
            acc += get_node_vars(old_u, equations, dg, k, l, old_element_id) *
                   forward_lower[i, k] * forward_upper[j, l]
        end
        set_node_vars!(u, acc, equations, dg, i, j, upper_left_id)
    end

    # Interpolate to upper right element
    for j in eachnode(dg), i in eachnode(dg)
        acc = zero(get_node_vars(u, equations, dg, i, j, element_id))
        for l in eachnode(dg), k in eachnode(dg)
            acc += get_node_vars(old_u, equations, dg, k, l, old_element_id) *
                   forward_upper[i, k] * forward_upper[j, l]
        end
        set_node_vars!(u, acc, equations, dg, i, j, upper_right_id)
    end

    return nothing
end

# Coarsen elements in the DG solver based on a list of cell_ids that should be removed.
# On `P4estMesh`, if an element coarsens the solution scaled by the Jacobian `J*u` is projected
# from the four children elements back onto the parent element. The solution on the parent
# element is then recovered by dividing by the new element Jacobian.
function coarsen!(u_ode::AbstractVector, adaptor,
                  mesh::Union{TreeMesh{2}, P4estMesh{2}},
                  equations, dg::DGSEM, cache, elements_to_remove)
    # Return early if there is nothing to do
    if isempty(elements_to_remove)
        if mpi_isparallel()
            # MPICache init uses all-to-all communication -> reinitialize even if there is nothing to do
            # locally (there still might be other MPI ranks that have coarsened elements)
            reinitialize_containers!(mesh, equations, dg, cache)
        end
        return
    end

    # Determine for each old element whether it needs to be removed
    to_be_removed = falses(nelements(dg, cache))
    to_be_removed[elements_to_remove] .= true

    # Retain current solution data and Jacobians
    old_n_elements = nelements(dg, cache)
    old_u_ode = copy(u_ode)
    old_inverse_jacobian = copy(cache.elements.inverse_jacobian)
    # OBS! If we don't GC.@preserve old_u_ode and old_inverse_jacobian, they might be GC'ed
    GC.@preserve old_u_ode old_inverse_jacobian begin
        old_u = wrap_array(old_u_ode, mesh, equations, dg, cache)

        if mesh isa P4estMesh
            # Loop over all elements in old container and scale the old solution by the Jacobian
            # prior to projection
            for old_element_id in 1:old_n_elements
                for v in eachvariable(equations)
                    old_u[v, .., old_element_id] .= (old_u[v, .., old_element_id] ./
                                                     old_inverse_jacobian[..,
                                                                          old_element_id])
                end
            end
        end

        reinitialize_containers!(mesh, equations, dg, cache)

        resize!(u_ode,
                nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache))
        u = wrap_array(u_ode, mesh, equations, dg, cache)

        # Loop over all elements in old container and either copy them or coarsen them
        skip = 0
        element_id = 1
        for old_element_id in 1:old_n_elements
            # If skip is non-zero, we just coarsened 2^ndims elements and need to omit the following elements
            if skip > 0
                skip -= 1
                continue
            end

            if to_be_removed[old_element_id]
                # If an element is to be removed, sanity check if the following elements
                # are also marked - otherwise there would be an error in the way the
                # cells/elements are sorted
                @assert all(to_be_removed[old_element_id:(old_element_id + 2^ndims(mesh) - 1)]) "bad cell/element order"

                # Coarsen elements and store solution directly in new data structure
                coarsen_elements!(u, element_id, old_u, old_element_id,
                                  adaptor, equations, dg)

                if mesh isa P4estMesh
                    # Before `element_id` is incremented, divide by the new Jacobian and save
                    # the result in the parent element
                    for v in eachvariable(equations)
                        u[v, .., element_id] .*= (4 .*
                                                  cache.elements.inverse_jacobian[..,
                                                                                  element_id])
                    end
                end

                # Increment `element_id` on the coarsened mesh by one and `skip` = 3 in 2D
                # because 4 children elements become 1 parent element
                element_id += 1
                skip = 2^ndims(mesh) - 1
            else
                if mesh isa P4estMesh
                    # Copy old element data to new element container and remove Jacobian scaling
                    for v in eachvariable(equations)
                        u[v, .., element_id] .= (old_u[v, .., old_element_id] .*
                                                 old_inverse_jacobian[..,
                                                                      old_element_id])
                    end
                else # isa TreeMesh
                    @views u[:, .., element_id] .= old_u[:, .., old_element_id]
                end
                # No coarsening occurred, so increment `element_id` on the new mesh by one
                element_id += 1
            end
        end
        # If everything is correct, we should have processed all elements.
        @assert element_id==nelements(dg, cache) + 1 "element_id = $element_id, nelements(dg, cache) = $(nelements(dg, cache))"
    end # GC.@preserve old_u_ode old_inverse_jacobian

    # Sanity check
    if mesh isa TreeMesh && isperiodic(mesh.tree) && nmortars(cache.mortars) == 0 &&
       !mpi_isparallel()
        @assert ninterfaces(cache.interfaces)==ndims(mesh) * nelements(dg, cache) ("For $(ndims(mesh))D and periodic domains and conforming elements, the number of interfaces must be $(ndims(mesh)) times the number of elements")
    end

    return nothing
end

function coarsen!(u_ode::AbstractVector, adaptor,
                  mesh::Union{TreeMesh{2}, P4estMesh{2}, TreeMesh{3}, P4estMesh{3}},
                  equations, dg::DGSEM, cache, cache_parabolic,
                  elements_to_remove)
    # Call `coarsen!` for the hyperbolic part, which does the heavy lifting of
    # actually transferring the solution to the coarsened cells
    coarsen!(u_ode, adaptor, mesh, equations, dg, cache, elements_to_remove)

    # Resize parabolic helper variables
    @unpack viscous_container = cache_parabolic
    resize!(viscous_container, equations, dg, cache)
    reinitialize_containers!(mesh, equations, dg, cache_parabolic)

    # Sanity check
    if mesh isa TreeMesh && isperiodic(mesh.tree) && nmortars(cache.mortars) == 0 &&
       !mpi_isparallel()
        @assert ninterfaces(cache_parabolic.interfaces)==ndims(mesh) *
                                                         nelements(dg, cache_parabolic) ("For $(ndims(mesh))D and periodic domains and conforming elements, the number of interfaces must be $(ndims(mesh)) times the number of elements")
    end

    return nothing
end

# TODO: Taal compare performance of different implementations
# Coarsen solution data u for four elements, using L2 projection
function coarsen_elements!(u::AbstractArray{<:Any, 4}, element_id,
                           old_u, old_element_id,
                           adaptor::LobattoLegendreAdaptorL2, equations, dg)
    @unpack reverse_upper, reverse_lower = adaptor

    # Store old element ids
    lower_left_id = old_element_id
    lower_right_id = old_element_id + 1
    upper_left_id = old_element_id + 2
    upper_right_id = old_element_id + 3

    @boundscheck begin
        @assert old_element_id >= 1
        @assert size(old_u, 1) == nvariables(equations)
        @assert size(old_u, 2) == nnodes(dg)
        @assert size(old_u, 3) == nnodes(dg)
        @assert size(old_u, 4) >= old_element_id + 3
        @assert element_id >= 1
        @assert size(u, 1) == nvariables(equations)
        @assert size(u, 2) == nnodes(dg)
        @assert size(u, 3) == nnodes(dg)
        @assert size(u, 4) >= element_id
    end

    for j in eachnode(dg), i in eachnode(dg)
        acc = zero(get_node_vars(u, equations, dg, i, j, element_id))

        # Project from lower left element
        for l in eachnode(dg), k in eachnode(dg)
            acc += get_node_vars(old_u, equations, dg, k, l, lower_left_id) *
                   reverse_lower[i, k] * reverse_lower[j, l]
        end

        # Project from lower right element
        for l in eachnode(dg), k in eachnode(dg)
            acc += get_node_vars(old_u, equations, dg, k, l, lower_right_id) *
                   reverse_upper[i, k] * reverse_lower[j, l]
        end

        # Project from upper left element
        for l in eachnode(dg), k in eachnode(dg)
            acc += get_node_vars(old_u, equations, dg, k, l, upper_left_id) *
                   reverse_lower[i, k] * reverse_upper[j, l]
        end

        # Project from upper right element
        for l in eachnode(dg), k in eachnode(dg)
            acc += get_node_vars(old_u, equations, dg, k, l, upper_right_id) *
                   reverse_upper[i, k] * reverse_upper[j, l]
        end

        # Update value
        set_node_vars!(u, acc, equations, dg, i, j, element_id)
    end
end

# Coarsen and refine elements in the DG solver based on a difference list.
function adapt!(u_ode::AbstractVector, adaptor, mesh::T8codeMesh{2}, equations,
                dg::DGSEM, cache, difference)

    # Return early if there is nothing to do.
    if !any(difference .!= 0)
        if mpi_isparallel()
            # MPICache init uses all-to-all communication -> reinitialize even if there is nothing to do
            # locally (there still might be other MPI ranks that have refined elements)
            reinitialize_containers!(mesh, equations, dg, cache)
        end
        return
    end

    # Number of (local) cells/elements.
    old_nelems = nelements(dg, cache)
    new_nelems = ncells(mesh)

    # Local element indices.
    old_index = 1
    new_index = 1

    # Note: This is true for `quads`.
    T8_CHILDREN = 4

    # Retain current solution and inverse Jacobian data.
    old_u_ode = copy(u_ode)
    old_inverse_jacobian = copy(cache.elements.inverse_jacobian)

    # OBS! If we don't GC.@preserve old_u_ode and old_inverse_jacobian, they might be GC'ed
    GC.@preserve old_u_ode begin
        old_u = wrap_array(old_u_ode, mesh, equations, dg, cache)

        # Loop over all elements in old container and scale the old solution by the Jacobian
        # prior to interpolation or projection
        for old_element_id in 1:old_nelems
            for v in eachvariable(equations)
                old_u[v, .., old_element_id] .= (old_u[v, .., old_element_id] ./
                                                 old_inverse_jacobian[..,
                                                                      old_element_id])
            end
        end

        reinitialize_containers!(mesh, equations, dg, cache)

        resize!(u_ode, nvariables(equations) * ndofs(mesh, dg, cache))
        u = wrap_array(u_ode, mesh, equations, dg, cache)

        while old_index <= old_nelems && new_index <= new_nelems
            if difference[old_index] > 0 # Refine.

                # Refine element and store solution directly in new data structure.
                refine_element!(u, new_index, old_u, old_index, adaptor, equations, dg)

                # Before indices are incremented divide by the new Jacobians on each
                # child element and save the result
                for m in 0:3 # loop over the children
                    for v in eachvariable(equations)
                        u[v, .., new_index + m] .*= (0.25f0 .*
                                                     cache.elements.inverse_jacobian[..,
                                                                                     new_index + m])
                    end
                end

                # Increment `old_index` on the original mesh and the `new_index`
                # on the refined mesh with the number of children, i.e., T8_CHILDREN = 4
                old_index += 1
                new_index += T8_CHILDREN

            elseif difference[old_index] < 0 # Coarsen.

                # If an element is to be removed, sanity check if the following elements
                # are also marked - otherwise there would be an error in the way the
                # cells/elements are sorted.
                @assert all(difference[old_index:(old_index + T8_CHILDREN - 1)] .< 0) "bad cell/element order"

                # Coarsen elements and store solution directly in new data structure.
                coarsen_elements!(u, new_index, old_u, old_index, adaptor, equations,
                                  dg)

                # Before the indices are incremented divide by the new Jacobian and save
                # the result again in the parent element
                for v in eachvariable(equations)
                    u[v, .., new_index] .*= (4 .* cache.elements.inverse_jacobian[..,
                                                                             new_index])
                end

                # Increment `old_index` on the original mesh with the number of children
                # (T8_CHILDREN = 4 in 2D) and the `new_index` by one for the single
                # coarsened element
                old_index += T8_CHILDREN
                new_index += 1

            else # No changes.

                # Copy old element data to new element container and remove Jacobian scaling
                for v in eachvariable(equations)
                    u[v, .., new_index] .= (old_u[v, .., old_index] .*
                                            old_inverse_jacobian[.., old_index])
                end

                # No refinement / coarsening occurred, so increment element index
                # on each mesh by one
                old_index += 1
                new_index += 1
            end
        end # while
    end # GC.@preserve old_u_ode old_inverse_jacobian

    return nothing
end

# this method is called when an `ControllerThreeLevel` is constructed
function create_cache(::Type{ControllerThreeLevel},
                      mesh::Union{TreeMesh{2}, P4estMesh{2}, T8codeMesh{2}}, equations,
                      dg::DG, cache)
    controller_value = Vector{Int}(undef, nelements(dg, cache))
    return (; controller_value)
end
end # @muladd
