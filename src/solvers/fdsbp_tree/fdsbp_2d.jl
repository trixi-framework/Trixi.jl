# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# General interface methods for SummationByPartsOperators.jl and Trixi.jl
# TODO: FD. Move to another file
nnodes(D::AbstractDerivativeOperator) = size(D, 1)
eachnode(D::AbstractDerivativeOperator) = Base.OneTo(nnodes(D))
get_nodes(D::AbstractDerivativeOperator) = grid(D)


# For dispatch
# TODO: FD. Move to another file
const FDSBP = DG{Basis} where {Basis<:AbstractDerivativeOperator}


# 2D containers
# TODO: FD. Move to another file
init_mortars(cell_ids, mesh, elements, mortar::Nothing) = nothing


# 2D caches
function create_cache(mesh::TreeMesh{2}, equations,
                      volume_integral::VolumeIntegralStrongForm, dg, uEltype)

  prototype = Array{SVector{nvariables(equations), uEltype}, ndims(mesh)}(
    undef, ntuple(_ -> nnodes(dg), ndims(mesh))...)
  f_threaded = [similar(prototype) for _ in 1:Threads.nthreads()]

  return (; f_threaded,)
end

create_cache(mesh, equations, mortar::Nothing, uEltype) = NamedTuple()
nmortars(mortar::Nothing) = 0


# 2D RHS
function calc_volume_integral!(du, u,
                               mesh::TreeMesh{2},
                               nonconservative_terms::Val{false}, equations,
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


function prolong2mortars!(cache, u, mesh, equations, mortar::Nothing,
                          surface_integral, dg::DG)
  @assert isempty(eachmortar(dg, cache))
end

function calc_mortar_flux!(surface_flux_values, mesh,
                           nonconservative_terms, equations,
                           mortar::Nothing,
                           surface_integral, dg::DG, cache)
  @assert isempty(eachmortar(dg, cache))
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


# AnalysisCallback
# TODO: FD. Move to another file
SolutionAnalyzer(D::AbstractDerivativeOperator) = D

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
                          dg::FDSBP, cache, cache_analysis; normalize=true)
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

  if normalize
    # For L2 error, divide by total volume
    total_volume_ = total_volume(mesh)
    l2_error = @. sqrt(l2_error / total_volume_)
  end

  return l2_error, linf_error
end


# TODO: FD. Visualization
#       We need to define better interfaces for all the plotting stuff.
#       Right now, the easiest solution is to use scatter plots such as
#       x = semi.cache.elements.node_coordinates[1, :, :, :] |> vec
#       y = semi.cache.elements.node_coordinates[2, :, :, :] |> vec
#       scatter(x, y, sol.u[end])


end # @muladd
