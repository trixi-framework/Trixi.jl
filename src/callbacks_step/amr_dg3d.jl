# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# Refine elements in the DG solver based on a list of cell_ids that should be refined
function refine!(u_ode::AbstractVector, adaptor,
                 mesh::Union{TreeMesh{3}, P4estMesh{3}},
                 equations, dg::DGSEM, cache, elements_to_refine)
  # Return early if there is nothing to do
  if isempty(elements_to_refine)
    return
  end

  # Determine for each existing element whether it needs to be refined
  needs_refinement = falses(nelements(dg, cache))
  needs_refinement[elements_to_refine] .= true

  # Retain current solution data
  old_n_elements = nelements(dg, cache)
  old_u_ode = copy(u_ode)
  GC.@preserve old_u_ode begin # OBS! If we don't GC.@preserve old_u_ode, it might be GC'ed
    old_u = wrap_array(old_u_ode, mesh, equations, dg, cache)

    reinitialize_containers!(mesh, equations, dg, cache)

    resize!(u_ode, nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache))
    u = wrap_array(u_ode, mesh, equations, dg, cache)

    # Loop over all elements in old container and either copy them or refine them
    u_tmp1 = Array{eltype(u), 4}(undef, nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg))
    u_tmp2 = Array{eltype(u), 4}(undef, nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg))
    element_id = 1
    for old_element_id in 1:old_n_elements
      if needs_refinement[old_element_id]
        # Refine element and store solution directly in new data structure
        refine_element!(u, element_id, old_u, old_element_id,
                        adaptor, equations, dg, u_tmp1, u_tmp2)
        element_id += 2^ndims(mesh)
      else
        # Copy old element data to new element container
        @views u[:, .., element_id] .= old_u[:, .., old_element_id]
        element_id += 1
      end
    end
    # If everything is correct, we should have processed all elements.
    # Depending on whether the last element processed above had to be refined or not,
    # the counter `element_id` can have two different values at the end.
    @assert element_id == nelements(dg, cache) + 1 || element_id == nelements(dg, cache) + 2^ndims(mesh) "element_id = $element_id, nelements(dg, cache) = $(nelements(dg, cache))"
  end # GC.@preserve old_u_ode

  # Sanity check
  if mesh isa TreeMesh && isperiodic(mesh.tree) && nmortars(cache.mortars) == 0
    @assert ninterfaces(cache.interfaces) == ndims(mesh) * nelements(dg, cache) ("For $(ndims(mesh))D and periodic domains and conforming elements, the number of interfaces must be $(ndims(mesh)) times the number of elements")
  end

  return nothing
end


# TODO: Taal compare performance of different implementations
# Refine solution data u for an element, using L2 projection (interpolation)
function refine_element!(u::AbstractArray{<:Any,5}, element_id,
                         old_u, old_element_id,
                         adaptor::LobattoLegendreAdaptorL2, equations, dg,
                         u_tmp1, u_tmp2)
  @unpack forward_upper, forward_lower = adaptor

  # Store new element ids
  bottom_lower_left_id  = element_id
  bottom_lower_right_id = element_id + 1
  bottom_upper_left_id  = element_id + 2
  bottom_upper_right_id = element_id + 3
  top_lower_left_id     = element_id + 4
  top_lower_right_id    = element_id + 5
  top_upper_left_id     = element_id + 6
  top_upper_right_id    = element_id + 7

  @boundscheck begin
    @assert old_element_id >= 1
    @assert size(old_u, 1) == nvariables(equations)
    @assert size(old_u, 2) == nnodes(dg)
    @assert size(old_u, 3) == nnodes(dg)
    @assert size(old_u, 4) == nnodes(dg)
    @assert size(old_u, 5) >= old_element_id
    @assert     element_id >= 1
    @assert size(    u, 1) == nvariables(equations)
    @assert size(    u, 2) == nnodes(dg)
    @assert size(    u, 3) == nnodes(dg)
    @assert size(    u, 4) == nnodes(dg)
    @assert size(    u, 5) >= element_id + 7
  end

  # Interpolate to bottom lower left element
  multiply_dimensionwise!(
    view(u,     :, :, :, :, bottom_lower_left_id), forward_lower, forward_lower, forward_lower,
    view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

  # Interpolate to bottom lower right element
  multiply_dimensionwise!(
    view(u,     :, :, :, :, bottom_lower_right_id), forward_upper, forward_lower, forward_lower,
    view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

  # Interpolate to bottom upper left element
  multiply_dimensionwise!(
    view(u,     :, :, :, :, bottom_upper_left_id), forward_lower, forward_upper, forward_lower,
    view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

  # Interpolate to bottom upper right element
  multiply_dimensionwise!(
    view(u,     :, :, :, :, bottom_upper_right_id), forward_upper, forward_upper, forward_lower,
    view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

  # Interpolate to top lower left element
  multiply_dimensionwise!(
    view(u,     :, :, :, :, top_lower_left_id), forward_lower, forward_lower, forward_upper,
    view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

  # Interpolate to top lower right element
  multiply_dimensionwise!(
    view(u,     :, :, :, :, top_lower_right_id), forward_upper, forward_lower, forward_upper,
    view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

  # Interpolate to top upper left element
  multiply_dimensionwise!(
    view(u,     :, :, :, :, top_upper_left_id), forward_lower, forward_upper, forward_upper,
    view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

  # Interpolate to top upper right element
  multiply_dimensionwise!(
    view(u,     :, :, :, :, top_upper_right_id), forward_upper, forward_upper, forward_upper,
    view(old_u, :, :, :, :, old_element_id), u_tmp1, u_tmp2)

  return nothing
end



# Coarsen elements in the DG solver based on a list of cell_ids that should be removed
function coarsen!(u_ode::AbstractVector, adaptor,
                  mesh::Union{TreeMesh{3}, P4estMesh{3}},
                  equations, dg::DGSEM, cache, elements_to_remove)
  # Return early if there is nothing to do
  if isempty(elements_to_remove)
    return
  end

  # Determine for each old element whether it needs to be removed
  to_be_removed = falses(nelements(dg, cache))
  to_be_removed[elements_to_remove] .= true

  # Retain current solution data
  old_n_elements = nelements(dg, cache)
  old_u_ode = copy(u_ode)
  GC.@preserve old_u_ode begin # OBS! If we don't GC.@preserve old_u_ode, it might be GC'ed
    old_u = wrap_array(old_u_ode, mesh, equations, dg, cache)

    reinitialize_containers!(mesh, equations, dg, cache)

    resize!(u_ode, nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache))
    u = wrap_array(u_ode, mesh, equations, dg, cache)

    # Loop over all elements in old container and either copy them or coarsen them
    u_tmp1 = Array{eltype(u), 4}(undef, nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg))
    u_tmp2 = Array{eltype(u), 4}(undef, nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg))
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
        @assert all(to_be_removed[old_element_id:(old_element_id+2^ndims(mesh)-1)]) "bad cell/element order"

        # Coarsen elements and store solution directly in new data structure
        coarsen_elements!(u, element_id, old_u, old_element_id,
                          adaptor, equations, dg, u_tmp1, u_tmp2)
        element_id += 1
        skip = 2^ndims(mesh) - 1
      else
        # Copy old element data to new element container
        @views u[:, .., element_id] .= old_u[:, .., old_element_id]
        element_id += 1
      end
    end
    # If everything is correct, we should have processed all elements.
    @assert element_id == nelements(dg, cache) + 1 "element_id = $element_id, nelements(dg, cache) = $(nelements(dg, cache))"
  end # GC.@preserve old_u_ode

  # Sanity check
  if mesh isa TreeMesh && isperiodic(mesh.tree) && nmortars(cache.mortars) == 0
    @assert ninterfaces(cache.interfaces) == ndims(mesh) * nelements(dg, cache) ("For $(ndims(mesh))D and periodic domains and conforming elements, the number of interfaces must be $(ndims(mesh)) times the number of elements")
  end

  return nothing
end


# TODO: Taal compare performance of different implementations
# Coarsen solution data u for four elements, using L2 projection
function coarsen_elements!(u::AbstractArray{<:Any,5}, element_id,
                           old_u, old_element_id,
                           adaptor::LobattoLegendreAdaptorL2, equations, dg,
                           u_tmp1, u_tmp2)
  @unpack reverse_upper, reverse_lower = adaptor

  # Store old element ids
  bottom_lower_left_id  = old_element_id
  bottom_lower_right_id = old_element_id + 1
  bottom_upper_left_id  = old_element_id + 2
  bottom_upper_right_id = old_element_id + 3
  top_lower_left_id     = old_element_id + 4
  top_lower_right_id    = old_element_id + 5
  top_upper_left_id     = old_element_id + 6
  top_upper_right_id    = old_element_id + 7

  @boundscheck begin
    @assert old_element_id >= 1
    @assert size(old_u, 1) == nvariables(equations)
    @assert size(old_u, 2) == nnodes(dg)
    @assert size(old_u, 3) == nnodes(dg)
    @assert size(old_u, 4) == nnodes(dg)
    @assert size(old_u, 5) >= old_element_id + 7
    @assert     element_id >= 1
    @assert size(    u, 1) == nvariables(equations)
    @assert size(    u, 2) == nnodes(dg)
    @assert size(    u, 3) == nnodes(dg)
    @assert size(    u, 4) == nnodes(dg)
    @assert size(    u, 5) >= element_id
  end

  # Project from bottom lower left element
  multiply_dimensionwise!(
    view(u,     :, :, :, :, element_id), reverse_lower, reverse_lower, reverse_lower,
    view(old_u, :, :, :, :, bottom_lower_left_id), u_tmp1, u_tmp2)

  # Project from bottom lower right element_variables
  add_multiply_dimensionwise!(
    view(u,     :, :, :, :, element_id), reverse_upper, reverse_lower, reverse_lower,
    view(old_u, :, :, :, :, bottom_lower_right_id), u_tmp1, u_tmp2)

  # Project from bottom upper left element
  add_multiply_dimensionwise!(
    view(u,     :, :, :, :, element_id), reverse_lower, reverse_upper, reverse_lower,
    view(old_u, :, :, :, :, bottom_upper_left_id), u_tmp1, u_tmp2)

  # Project from bottom upper right element
  add_multiply_dimensionwise!(
    view(u,     :, :, :, :, element_id), reverse_upper, reverse_upper, reverse_lower,
    view(old_u, :, :, :, :, bottom_upper_right_id), u_tmp1, u_tmp2)

  # Project from top lower left element
  add_multiply_dimensionwise!(
    view(u,     :, :, :, :, element_id), reverse_lower, reverse_lower, reverse_upper,
    view(old_u, :, :, :, :, top_lower_left_id), u_tmp1, u_tmp2)

  # Project from top lower right element
  add_multiply_dimensionwise!(
    view(u,     :, :, :, :, element_id), reverse_upper, reverse_lower, reverse_upper,
    view(old_u, :, :, :, :, top_lower_right_id), u_tmp1, u_tmp2)

  # Project from top upper left element
  add_multiply_dimensionwise!(
    view(u,     :, :, :, :, element_id), reverse_lower, reverse_upper, reverse_upper,
    view(old_u, :, :, :, :, top_upper_left_id), u_tmp1, u_tmp2)

  # Project from top upper right element
  add_multiply_dimensionwise!(
    view(u,     :, :, :, :, element_id), reverse_upper, reverse_upper, reverse_upper,
    view(old_u, :, :, :, :, top_upper_right_id), u_tmp1, u_tmp2)

  return nothing
end


# this method is called when an `ControllerThreeLevel` is constructed
function create_cache(::Type{ControllerThreeLevel},
                      mesh::Union{TreeMesh{3}, P4estMesh{3}},
                      equations, dg::DG, cache)

  controller_value = Vector{Int}(undef, nelements(dg, cache))
  return (; controller_value)
end


end # @muladd
