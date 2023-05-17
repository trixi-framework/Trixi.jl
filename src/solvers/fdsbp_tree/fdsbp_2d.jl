# !!! warning "Experimental implementation (upwind SBP)"
#     This is an experimental feature and may change in future releases.

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# 2D caches
function create_cache(mesh::TreeMesh{2}, equations,
                      volume_integral::VolumeIntegralStrongForm, dg, uEltype)

  prototype = Array{SVector{nvariables(equations), uEltype}, ndims(mesh)}(
    undef, ntuple(_ -> nnodes(dg), ndims(mesh))...)
  f_threaded = [similar(prototype) for _ in 1:Threads.nthreads()]

  return (; f_threaded,)
end

function create_cache(mesh::TreeMesh{2}, equations,
                      volume_integral::VolumeIntegralUpwind, dg, uEltype)

  u_node = SVector{nvariables(equations), uEltype}(ntuple(_ -> zero(uEltype), Val{nvariables(equations)}()))
  f = StructArray([(u_node, u_node)])
  f_minus_plus_threaded = [similar(f, ntuple(_ -> nnodes(dg), ndims(mesh))...) for _ in 1:Threads.nthreads()]

  f_minus, f_plus = StructArrays.components(f_minus_plus_threaded[1])
  f_minus_threaded = [f_minus]
  f_plus_threaded = [f_plus]
  for i in 2:Threads.nthreads()
    f_minus, f_plus = StructArrays.components(f_minus_plus_threaded[i])
    push!(f_minus_threaded, f_minus)
    push!(f_plus_threaded, f_plus)
  end

  return (; f_minus_plus_threaded, f_minus_threaded, f_plus_threaded,)
end


# 2D volume integral contributions for `VolumeIntegralStrongForm`
function calc_volume_integral!(du, u,
                               mesh::TreeMesh{2},
                               nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralStrongForm,
                               dg::FDSBP, cache)
  D = dg.basis # SBP derivative operator
  @unpack f_threaded = cache

  # SBP operators from SummationByPartsOperators.jl implement the basic interface
  # of matrix-vector multiplication. Thus, we pass an "array of structures",
  # packing all variables per node in an `SVector`.
  if nvariables(equations) == 1
    # `reinterpret(reshape, ...)` removes the leading dimension only if more
    # than one variable is used.
    u_vectors  = reshape(reinterpret(SVector{nvariables(equations), eltype(u)}, u),
                         nnodes(dg), nnodes(dg), nelements(dg, cache))
    du_vectors = reshape(reinterpret(SVector{nvariables(equations), eltype(du)}, du),
                         nnodes(dg), nnodes(dg), nelements(dg, cache))
  else
    u_vectors  = reinterpret(reshape, SVector{nvariables(equations), eltype(u)}, u)
    du_vectors = reinterpret(reshape, SVector{nvariables(equations), eltype(du)}, du)
  end

  # Use the tensor product structure to compute the discrete derivatives of
  # the fluxes line-by-line and add them to `du` for each element.
  @threaded for element in eachelement(dg, cache)
    f_element = f_threaded[Threads.threadid()]
    u_element = view(u_vectors,  :, :, element)

    # x direction
    @. f_element = flux(u_element, 1, equations)
    for j in eachnode(dg)
      mul!(view(du_vectors, :, j, element), D, view(f_element, :, j),
           one(eltype(du)), one(eltype(du)))
    end

    # y direction
    @. f_element = flux(u_element, 2, equations)
    for i in eachnode(dg)
      mul!(view(du_vectors, i, :, element), D, view(f_element, i, :),
           one(eltype(du)), one(eltype(du)))
    end
  end

  return nothing
end


# 2D volume integral contributions for `VolumeIntegralUpwind`.
# Note that the plus / minus notation of the operators does not refer to the
# upwind / downwind directions of the fluxes.
# Instead, the plus / minus refers to the direction of the biasing within
# the finite difference stencils. Thus, the D^- operator acts on the positive
# part of the flux splitting f^+ and the D^+ operator acts on the negative part
# of the flux splitting f^-.
function calc_volume_integral!(du, u,
                               mesh::TreeMesh{2},
                               nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralUpwind,
                               dg::FDSBP, cache)
  # Assume that
  # dg.basis isa SummationByPartsOperators.UpwindOperators
  D_minus = dg.basis.minus # Upwind SBP D^- derivative operator
  D_plus = dg.basis.plus   # Upwind SBP D^+ derivative operator
  @unpack f_minus_plus_threaded, f_minus_threaded, f_plus_threaded = cache
  @unpack splitting = volume_integral

  # SBP operators from SummationByPartsOperators.jl implement the basic interface
  # of matrix-vector multiplication. Thus, we pass an "array of structures",
  # packing all variables per node in an `SVector`.
  if nvariables(equations) == 1
    # `reinterpret(reshape, ...)` removes the leading dimension only if more
    # than one variable is used.
    u_vectors  = reshape(reinterpret(SVector{nvariables(equations), eltype(u)}, u),
                         nnodes(dg), nnodes(dg), nelements(dg, cache))
    du_vectors = reshape(reinterpret(SVector{nvariables(equations), eltype(du)}, du),
                         nnodes(dg), nnodes(dg), nelements(dg, cache))
  else
    u_vectors  = reinterpret(reshape, SVector{nvariables(equations), eltype(u)}, u)
    du_vectors = reinterpret(reshape, SVector{nvariables(equations), eltype(du)}, du)
  end

  # Use the tensor product structure to compute the discrete derivatives of
  # the fluxes line-by-line and add them to `du` for each element.
  @threaded for element in eachelement(dg, cache)
    # f_minus_plus_element wraps the storage provided by f_minus_element and
    # f_plus_element such that we can use a single plain broadcasting below.
    # f_minus_element and f_plus_element are updated in broadcasting calls
    # of the form `@. f_minus_plus_element = ...`.
    f_minus_plus_element = f_minus_plus_threaded[Threads.threadid()]
    f_minus_element = f_minus_threaded[Threads.threadid()]
    f_plus_element = f_plus_threaded[Threads.threadid()]
    u_element = view(u_vectors, :, :, element)

    # x direction
    @. f_minus_plus_element = splitting(u_element, 1, equations)
    for j in eachnode(dg)
      mul!(view(du_vectors, :, j, element), D_minus, view(f_plus_element, :, j),
           one(eltype(du)), one(eltype(du)))
      mul!(view(du_vectors, :, j, element), D_plus, view(f_minus_element, :, j),
           one(eltype(du)), one(eltype(du)))
    end

    # y direction
    @. f_minus_plus_element = splitting(u_element, 2, equations)
    for i in eachnode(dg)
      mul!(view(du_vectors, i, :, element), D_minus, view(f_plus_element, i, :),
           one(eltype(du)), one(eltype(du)))
      mul!(view(du_vectors, i, :, element), D_plus, view(f_minus_element, i, :),
           one(eltype(du)), one(eltype(du)))
    end
  end

  return nothing
end


function calc_surface_integral!(du, u, mesh::TreeMesh{2},
                                equations, surface_integral::SurfaceIntegralStrongForm,
                                dg::DG, cache)
  inv_weight_left  = inv(left_boundary_weight(dg.basis))
  inv_weight_right = inv(right_boundary_weight(dg.basis))
  @unpack surface_flux_values = cache.elements

  @threaded for element in eachelement(dg, cache)
    for l in eachnode(dg)
      # surface at -x
      u_node = get_node_vars(u, equations, dg, 1, l, element)
      f_node = flux(u_node, 1, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 1, element)
      multiply_add_to_node_vars!(du, inv_weight_left, -(f_num - f_node),
                                 equations, dg, 1, l, element)

      # surface at +x
      u_node = get_node_vars(u, equations, dg, nnodes(dg), l, element)
      f_node = flux(u_node, 1, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 2, element)
      multiply_add_to_node_vars!(du, inv_weight_right, +(f_num - f_node),
                                 equations, dg, nnodes(dg), l, element)

      # surface at -y
      u_node = get_node_vars(u, equations, dg, l, 1, element)
      f_node = flux(u_node, 2, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 3, element)
      multiply_add_to_node_vars!(du, inv_weight_left, -(f_num - f_node),
                                 equations, dg, l, 1, element)

      # surface at +y
      u_node = get_node_vars(u, equations, dg, l, nnodes(dg), element)
      f_node = flux(u_node, 2, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 4, element)
      multiply_add_to_node_vars!(du, inv_weight_right, +(f_num - f_node),
                                 equations, dg, l, nnodes(dg), element)
    end
  end

  return nothing
end


# Specialized interface flux computation because the upwind solver does
# not require a standard numerical flux (Riemann solver). The flux splitting
# already separates the solution information into right-traveling and
# left-traveling information. So we only need to compute the appropriate
# flux information at each side of an interface.
function calc_interface_flux!(surface_flux_values,
                              mesh::TreeMesh{2},
                              nonconservative_terms::False, equations,
                              surface_integral::SurfaceIntegralUpwind,
                              dg::FDSBP, cache)
  @unpack splitting = surface_integral
  @unpack u, neighbor_ids, orientations = cache.interfaces

  @threaded for interface in eachinterface(dg, cache)
    # Get neighboring elements
    left_id  = neighbor_ids[1, interface]
    right_id = neighbor_ids[2, interface]

    # Determine interface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    # orientation = 2: left -> 4, right -> 3
    left_direction  = 2 * orientations[interface]
    right_direction = 2 * orientations[interface] - 1

    for i in eachnode(dg)
      # Pull the left and right solution data
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)

      # Compute the upwind coupling terms where right-traveling
      # information comes from the left and left-traveling information
      # comes from the right
      flux_minus_rr = splitting(u_rr, Val{:minus}(), orientations[interface], equations)
      flux_plus_ll  = splitting(u_ll, Val{:plus}(),  orientations[interface], equations)

      # Save the upwind coupling into the appropriate side of the elements
      for v in eachvariable(equations)
        surface_flux_values[v, i, left_direction,  left_id]  = flux_minus_rr[v]
        surface_flux_values[v, i, right_direction, right_id] = flux_plus_ll[v]
      end
    end
  end

  return nothing
end


# Implementation of fully upwind SATs. The surface flux values are pre-computed
# in the specialized `calc_interface_flux` routine. These SATs are still of
# a strong form penalty type, except that the interior flux at a particular
# side of the element are computed in the upwind direction.
function calc_surface_integral!(du, u, mesh::TreeMesh{2},
                                equations, surface_integral::SurfaceIntegralUpwind,
                                dg::FDSBP, cache)
  inv_weight_left  = inv(left_boundary_weight(dg.basis))
  inv_weight_right = inv(right_boundary_weight(dg.basis))
  @unpack surface_flux_values = cache.elements
  @unpack splitting = surface_integral


  @threaded for element in eachelement(dg, cache)
    for l in eachnode(dg)
      # surface at -x
      u_node = get_node_vars(u, equations, dg, 1, l, element)
      f_node = splitting(u_node, Val{:plus}(), 1, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 1, element)
      multiply_add_to_node_vars!(du, inv_weight_left, -(f_num - f_node),
                                 equations, dg, 1, l, element)

      # surface at +x
      u_node = get_node_vars(u, equations, dg, nnodes(dg), l, element)
      f_node = splitting(u_node, Val{:minus}(), 1, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 2, element)
      multiply_add_to_node_vars!(du, inv_weight_right, +(f_num - f_node),
                                 equations, dg, nnodes(dg), l, element)

      # surface at -y
      u_node = get_node_vars(u, equations, dg, l, 1, element)
      f_node = splitting(u_node, Val{:plus}(), 2, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 3, element)
      multiply_add_to_node_vars!(du, inv_weight_left, -(f_num - f_node),
                                 equations, dg, l, 1, element)

      # surface at +y
      u_node = get_node_vars(u, equations, dg, l, nnodes(dg), element)
      f_node = splitting(u_node, Val{:minus}(), 2, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 4, element)
      multiply_add_to_node_vars!(du, inv_weight_right, +(f_num - f_node),
                                 equations, dg, l, nnodes(dg), element)
    end
  end

  return nothing
end


# AnalysisCallback
function integrate_via_indices(func::Func, u,
                               mesh::TreeMesh{2}, equations,
                               dg::FDSBP, cache, args...; normalize=true) where {Func}
  # TODO: FD. This is rather inefficient right now and allocates...
  weights = diag(SummationByPartsOperators.mass_matrix(dg.basis))

  # Initialize integral with zeros of the right shape
  integral = zero(func(u, 1, 1, 1, equations, dg, args...))

  # Use quadrature to numerically integrate over entire domain
  for element in eachelement(dg, cache)
    volume_jacobian_ = volume_jacobian(element, mesh, cache)
    for j in eachnode(dg), i in eachnode(dg)
      integral += volume_jacobian_ * weights[i] * weights[j] * func(u, i, j, element, equations, dg, args...)
    end
  end

  # Normalize with total volume
  if normalize
    integral = integral / total_volume(mesh)
  end

  return integral
end

function calc_error_norms(func, u, t, analyzer,
                          mesh::TreeMesh{2}, equations, initial_condition,
                          dg::FDSBP, cache, cache_analysis)
  # TODO: FD. This is rather inefficient right now and allocates...
  weights = diag(SummationByPartsOperators.mass_matrix(dg.basis))
  @unpack node_coordinates = cache.elements

  # Set up data structures
  l2_error   = zero(func(get_node_vars(u, equations, dg, 1, 1, 1), equations))
  linf_error = copy(l2_error)

  # Iterate over all elements for error calculations
  for element in eachelement(dg, cache)
    # Calculate errors at each node
    volume_jacobian_ = volume_jacobian(element, mesh, cache)

    for j in eachnode(analyzer), i in eachnode(analyzer)
      u_exact = initial_condition(
        get_node_coords(node_coordinates, equations, dg, i, j, element), t, equations)
      diff = func(u_exact, equations) - func(
        get_node_vars(u, equations, dg, i, j, element), equations)
      l2_error += diff.^2 * (weights[i] * weights[j] * volume_jacobian_)
      linf_error = @. max(linf_error, abs(diff))
    end
  end

  # For L2 error, divide by total volume
  total_volume_ = total_volume(mesh)
  l2_error = @. sqrt(l2_error / total_volume_)

  return l2_error, linf_error
end


end # @muladd
