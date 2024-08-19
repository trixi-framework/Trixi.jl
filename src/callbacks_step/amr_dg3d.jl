# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Refine elements in the DG solver based on a list of cell_ids that should be refined.
# On `P4estMesh`, if an element refines the solution scaled by the Jacobian `J*u` is interpolated
# from the parent element into the eight children elements. The solution on each child
# element is then recovered by dividing by the new element Jacobians.
function refine!(u_ode::AbstractVector, adaptor, mesh::Union{TreeMesh{3}, P4estMesh{3}},
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
        u_tmp1 = Array{eltype(u), 4}(undef, nvariables(equations), nnodes(dg),
                                     nnodes(dg), nnodes(dg))
        u_tmp2 = Array{eltype(u), 4}(undef, nvariables(equations), nnodes(dg),
                                     nnodes(dg), nnodes(dg))
        element_id = 1
        for old_element_id in 1:old_n_elements
            if needs_refinement[old_element_id]
                # Refine element and store solution directly in new data structure
                refine_element!(u, element_id, old_u, old_element_id, adaptor,
                                equations, dg, u_tmp1, u_tmp2)

                if mesh isa P4estMesh
                    # Before `element_id` is incremented, divide by the new Jacobians on each
                    # child element and save the result
                    for m in 0:7 # loop over the children
                        for v in eachvariable(equations)
                            u[v, .., element_id + m] .*= (0.125f0 .*
                                                          cache.elements.inverse_jacobian[..,
                                                                                          element_id + m])
                        end
                    end
                end

                # Increment `element_id` on the refined mesh with the number
                # of children, i.e., 8 in 3D
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
    if mesh isa TreeMesh && isperiodic(mesh.tree) && nmortars(cache.mortars) == 0
        @assert ninterfaces(cache.interfaces)==ndims(mesh) * nelements(dg, cache) ("For $(ndims(mesh))D and periodic domains and conforming elements, the number of interfaces must be $(ndims(mesh)) times the number of elements")
    end

    return nothing
end

# TODO: Taal compare performance of different implementations
# Refine solution data u for an element, using L2 projection (interpolation)
function refine_element!(u::AbstractArray{<:Any, 5}, element_id,
                         old_u, old_element_id,
                         adaptor::LobattoLegendreAdaptorL2, equations, dg,
                         u_tmp1, u_tmp2)
    @unpack forward_upper, forward_lower = adaptor

    # Store new element ids
    bottom_lower_left_id = element_id
    bottom_lower_right_id = element_id + 1
    bottom_upper_left_id = element_id + 2
    bottom_upper_right_id = element_id + 3
    top_lower_left_id = element_id + 4
    top_lower_right_id = element_id + 5
    top_upper_left_id = element_id + 6
    top_upper_right_id = element_id + 7

    @boundscheck begin
        @assert old_element_id >= 1
        @assert size(old_u, 1) == nvariables(equations)
        @assert size(old_u, 2) == nnodes(dg)
        @assert size(old_u, 3) == nnodes(dg)
        @assert size(old_u, 4) == nnodes(dg)
        @assert size(old_u, 5) >= old_element_id
        @assert element_id >= 1
        @assert size(u, 1) == nvariables(equations)
        @assert size(u, 2) == nnodes(dg)
        @assert size(u, 3) == nnodes(dg)
        @assert size(u, 4) == nnodes(dg)
        @assert size(u, 5) >= element_id + 7
    end

    # Interpolate to bottom lower left element
    multiply_dimensionwise!(view(u, :, :, :, :, bottom_lower_left_id), forward_lower,
                            forward_lower, forward_lower,
                            view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # Interpolate to bottom lower right element
    multiply_dimensionwise!(view(u, :, :, :, :, bottom_lower_right_id), forward_upper,
                            forward_lower, forward_lower,
                            view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # Interpolate to bottom upper left element
    multiply_dimensionwise!(view(u, :, :, :, :, bottom_upper_left_id), forward_lower,
                            forward_upper, forward_lower,
                            view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # Interpolate to bottom upper right element
    multiply_dimensionwise!(view(u, :, :, :, :, bottom_upper_right_id), forward_upper,
                            forward_upper, forward_lower,
                            view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # Interpolate to top lower left element
    multiply_dimensionwise!(view(u, :, :, :, :, top_lower_left_id), forward_lower,
                            forward_lower, forward_upper,
                            view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # Interpolate to top lower right element
    multiply_dimensionwise!(view(u, :, :, :, :, top_lower_right_id), forward_upper,
                            forward_lower, forward_upper,
                            view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # Interpolate to top upper left element
    multiply_dimensionwise!(view(u, :, :, :, :, top_upper_left_id), forward_lower,
                            forward_upper, forward_upper,
                            view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    # Interpolate to top upper right element
    multiply_dimensionwise!(view(u, :, :, :, :, top_upper_right_id), forward_upper,
                            forward_upper, forward_upper,
                            view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

    return nothing
end

# Coarsen elements in the DG solver based on a list of cell_ids that should be removed.
# On `P4estMesh`, if an element coarsens the solution scaled by the Jacobian `J*u` is projected
# from the eight children elements back onto the parent element. The solution on the parent
# element is then recovered by dividing by the new element Jacobian.
function coarsen!(u_ode::AbstractVector, adaptor,
                  mesh::Union{TreeMesh{3}, P4estMesh{3}},
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
        u_tmp1 = Array{eltype(u), 4}(undef, nvariables(equations), nnodes(dg),
                                     nnodes(dg), nnodes(dg))
        u_tmp2 = Array{eltype(u), 4}(undef, nvariables(equations), nnodes(dg),
                                     nnodes(dg), nnodes(dg))
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
                coarsen_elements!(u, element_id, old_u, old_element_id, adaptor,
                                  equations, dg, u_tmp1, u_tmp2)

                if mesh isa P4estMesh
                    # Before `element_id` is incremented, divide by the new Jacobian and save
                    # the result in the parent element
                    for v in eachvariable(equations)
                        u[v, .., element_id] .*= (8 .*
                                                  cache.elements.inverse_jacobian[..,
                                                                                  element_id])
                    end
                end

                # Increment `element_id` on the coarsened mesh by one and `skip` = 7 in 3D
                # because 8 children elements become 1 parent element
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
    if mesh isa TreeMesh && isperiodic(mesh.tree) && nmortars(cache.mortars) == 0
        @assert ninterfaces(cache.interfaces)==ndims(mesh) * nelements(dg, cache) ("For $(ndims(mesh))D and periodic domains and conforming elements, the number of interfaces must be $(ndims(mesh)) times the number of elements")
    end

    return nothing
end

# TODO: Taal compare performance of different implementations
# Coarsen solution data u for four elements, using L2 projection
function coarsen_elements!(u::AbstractArray{<:Any, 5}, element_id,
                           old_u, old_element_id,
                           adaptor::LobattoLegendreAdaptorL2, equations, dg,
                           u_tmp1, u_tmp2)
    @unpack reverse_upper, reverse_lower = adaptor

    # Store old element ids
    bottom_lower_left_id = old_element_id
    bottom_lower_right_id = old_element_id + 1
    bottom_upper_left_id = old_element_id + 2
    bottom_upper_right_id = old_element_id + 3
    top_lower_left_id = old_element_id + 4
    top_lower_right_id = old_element_id + 5
    top_upper_left_id = old_element_id + 6
    top_upper_right_id = old_element_id + 7

    @boundscheck begin
        @assert old_element_id >= 1
        @assert size(old_u, 1) == nvariables(equations)
        @assert size(old_u, 2) == nnodes(dg)
        @assert size(old_u, 3) == nnodes(dg)
        @assert size(old_u, 4) == nnodes(dg)
        @assert size(old_u, 5) >= old_element_id + 7
        @assert element_id >= 1
        @assert size(u, 1) == nvariables(equations)
        @assert size(u, 2) == nnodes(dg)
        @assert size(u, 3) == nnodes(dg)
        @assert size(u, 4) == nnodes(dg)
        @assert size(u, 5) >= element_id
    end

    # Project from bottom lower left element
    multiply_dimensionwise!(view(u, :, :, :, :, element_id), reverse_lower,
                            reverse_lower, reverse_lower,
                            view(old_u, :, :, :, :, bottom_lower_left_id), u_tmp1,
                            u_tmp2)

    # Project from bottom lower right element_variables
    add_multiply_dimensionwise!(view(u, :, :, :, :, element_id), reverse_upper,
                                reverse_lower, reverse_lower,
                                view(old_u, :, :, :, :, bottom_lower_right_id), u_tmp1,
                                u_tmp2)

    # Project from bottom upper left element
    add_multiply_dimensionwise!(view(u, :, :, :, :, element_id), reverse_lower,
                                reverse_upper, reverse_lower,
                                view(old_u, :, :, :, :, bottom_upper_left_id), u_tmp1,
                                u_tmp2)

    # Project from bottom upper right element
    add_multiply_dimensionwise!(view(u, :, :, :, :, element_id), reverse_upper,
                                reverse_upper, reverse_lower,
                                view(old_u, :, :, :, :, bottom_upper_right_id), u_tmp1,
                                u_tmp2)

    # Project from top lower left element
    add_multiply_dimensionwise!(view(u, :, :, :, :, element_id), reverse_lower,
                                reverse_lower, reverse_upper,
                                view(old_u, :, :, :, :, top_lower_left_id), u_tmp1,
                                u_tmp2)

    # Project from top lower right element
    add_multiply_dimensionwise!(view(u, :, :, :, :, element_id), reverse_upper,
                                reverse_lower, reverse_upper,
                                view(old_u, :, :, :, :, top_lower_right_id), u_tmp1,
                                u_tmp2)

    # Project from top upper left element
    add_multiply_dimensionwise!(view(u, :, :, :, :, element_id), reverse_lower,
                                reverse_upper, reverse_upper,
                                view(old_u, :, :, :, :, top_upper_left_id), u_tmp1,
                                u_tmp2)

    # Project from top upper right element
    add_multiply_dimensionwise!(view(u, :, :, :, :, element_id), reverse_upper,
                                reverse_upper, reverse_upper,
                                view(old_u, :, :, :, :, top_upper_right_id), u_tmp1,
                                u_tmp2)

    return nothing
end

# this method is called when an `ControllerThreeLevel` is constructed
function create_cache(::Type{ControllerThreeLevel},
                      mesh::Union{TreeMesh{3}, P4estMesh{3}, T8codeMesh{3}},
                      equations, dg::DG, cache)
    controller_value = Vector{Int}(undef, nelements(dg, cache))
    return (; controller_value)
end

# Coarsen and refine elements in the DG solver based on a difference list.
function adapt!(u_ode::AbstractVector, adaptor, mesh::T8codeMesh{3}, equations,
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

    # Note: This is only true for `hexs`.
    T8_CHILDREN = 8

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

        u_tmp1 = Array{eltype(u), 4}(undef, nvariables(equations), nnodes(dg),
                                     nnodes(dg), nnodes(dg))
        u_tmp2 = Array{eltype(u), 4}(undef, nvariables(equations), nnodes(dg),
                                     nnodes(dg), nnodes(dg))

        while old_index <= old_nelems && new_index <= new_nelems
            if difference[old_index] > 0 # Refine.

                # Refine element and store solution directly in new data structure.
                refine_element!(u, new_index, old_u, old_index, adaptor, equations, dg,
                                u_tmp1, u_tmp2)

                # Before indices are incremented divide by the new Jacobians on each
                # child element and save the result
                for m in 0:7 # loop over the children
                    for v in eachvariable(equations)
                        u[v, .., new_index + m] .*= (0.125f0 .*
                                                     cache.elements.inverse_jacobian[..,
                                                                                     new_index + m])
                    end
                end

                # Increment `old_index` on the original mesh and the `new_index`
                # on the refined mesh with the number of children, i.e., T8_CHILDREN = 8
                old_index += 1
                new_index += T8_CHILDREN

            elseif difference[old_index] < 0 # Coarsen.

                # If an element is to be removed, sanity check if the following elements
                # are also marked - otherwise there would be an error in the way the
                # cells/elements are sorted.
                @assert all(difference[old_index:(old_index + T8_CHILDREN - 1)] .< 0) "bad cell/element order"

                # Coarsen elements and store solution directly in new data structure.
                coarsen_elements!(u, new_index, old_u, old_index, adaptor, equations,
                                  dg, u_tmp1, u_tmp2)

                # Before the indices are incremented divide by the new Jacobian and save
                # the result again in the parent element
                for v in eachvariable(equations)
                    u[v, .., new_index] .*= (8 .* cache.elements.inverse_jacobian[..,
                                                                             new_index])
                end

                # Increment `old_index` on the original mesh with the number of children
                # (T8_CHILDREN = 8 in 3D) and the `new_index` by one for the single
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
end # @muladd
