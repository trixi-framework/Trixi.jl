# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function limiter_shallow_water!(u, threshold::Real, variable,
                            mesh::AbstractMesh{1}, equations::ShallowWaterEquations1D,
                            dg::DGSEM, cache)
  @unpack weights = dg.basis

  @threaded for element in eachelement(dg, cache)
    # determine minimum value
    value_min = typemax(eltype(u))
    for i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, element)
      value_min = min(value_min, variable(u_node, equations))
    end

    # detect if limiting is necessary
    value_min < threshold || continue

    # compute mean value
    u_mean = zero(get_node_vars(u, equations, dg, 1, element))
    for i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, element)
      u_mean += u_node * weights[i]
    end
    # note that the reference element is [-1,1]^ndims(dg), thus the weights sum to 2
    u_mean = u_mean / 2^ndims(mesh)

    # We compute the value directly with the mean values, as we assume that
    # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
    value_mean = variable(u_mean, equations)
    theta = (value_mean - threshold) / (value_mean - value_min)
    for i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, element)

      # Cut off velocity in case that the waterheight is smaller than the threshold

      h_node, h_v_node, b_node = u_node
      h_mean, h_v_mean, _ = u_mean # b_mean is not used as b_node must not be overwritten

      # Set them both to zero to apply linear combination correctly
      h_v_node = h_v_node * Int32(h_node > threshold)

      h_v_mean = h_v_mean * Int32(h_node > threshold)

      u_node = SVector(h_node, h_v_node, b_node)
      u_mean = SVector(h_mean, h_v_mean, zero(eltype(u)))

      # When velocity is cut off, the only averaged value is the waterheight,
      # because the velocity is set to zero and this value is passed.
      # Otherwise, the velocity is averaged, as well.
      # Note that the auxiliary bottom topography variable `b` is never limited.
      set_node_vars!(u, theta * u_node + (1-theta) * u_mean,
                     (1,2), dg, i, element)
    end
  end

  # An extra "safety" check is done over all the degrees of
  # freedom after the limiting in order to avoid dry nodes.
  # If the value_mean < threshold before applying limiter, there
  # could still be dry nodes afterwards due to logic of limiter.
  @threaded for element in eachelement(dg, cache)

    for i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, element)

      h, hv, b = u_node

      h = h * Int32(h > threshold) + threshold * Int32(h <= threshold)
      hv = hv * Int32(h > threshold)

      u_node = SVector(h, hv, b)

      set_node_vars!(u, u_node, equations, dg, i, element)
    end
  end

  return nothing
end


end # @muladd
