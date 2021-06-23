
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
      surface_contribution = -(f_num - f_node) * inv_weight_left
      add_to_node_vars!(du, surface_contribution, equations, dg, 1, l, element)

      # surface at +x
      u_node = get_node_vars(u, equations, dg, nnodes(dg), l, element)
      f_node = flux(u_node, 1, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 2, element)
      surface_contribution = +(f_num - f_node) * inv_weight_right
      add_to_node_vars!(du, surface_contribution, equations, dg, nnodes(dg), l, element)

      # surface at -y
      u_node = get_node_vars(u, equations, dg, l, 1, element)
      f_node = flux(u_node, 2, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 3, element)
      surface_contribution = -(f_num - f_node) * inv_weight_left
      add_to_node_vars!(du, surface_contribution, equations, dg, l, 1, element)

      # surface at +y
      u_node = get_node_vars(u, equations, dg, l, nnodes(dg), element)
      f_node = flux(u_node, 2, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 4, element)
      surface_contribution = +(f_num - f_node) * inv_weight_right
      add_to_node_vars!(du, surface_contribution, equations, dg, l, nnodes(dg), element)
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


# TODO: FD. Visualization
#       We need to define better interfaces for all the plotting stuff.
#       Right now, the easiest solution is to use scatter plots such as
#       x = semi.cache.elements.node_coordinates[1, :, :, :] |> vec
#       y = semi.cache.elements.node_coordinates[2, :, :, :] |> vec
#       scatter(x, y, sol.u[end])


# Special rhs evaluations for the incompressible Euler equations
function rhs!(du, u, t,
              mesh::TreeMesh{2},
              equations::IncompressibleEulerEquations2D,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @trixi_timeit timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @trixi_timeit timer() "volume integral" calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  # # Prolong solution to interfaces
  # @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
  #   cache, u, mesh, equations, dg.surface_integral, dg)
  #
  # # Prolong solution to boundaries
  # @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
  #   cache, u, mesh, equations, dg.surface_integral, dg)

  # Calculate surface integrals
  @trixi_timeit timer() "surface integral" calc_surface_integral!(
    du, u, mesh, equations, dg.surface_integral, dg, cache)

  # Apply Jacobian from mapping to reference element
  @trixi_timeit timer() "Jacobian" apply_jacobian!(
    du, mesh, equations, dg, cache)

  # Calculate source terms
  @trixi_timeit timer() "source terms" calc_sources!(
    du, u, t, source_terms, equations, dg, cache)

  return nothing
end


# Special volume integral for the incompressible Euler equations
function calc_volume_integral!(du, u,
                               mesh::TreeMesh{2},
                               nonconservative_terms::Val{false},
                               equations::IncompressibleEulerEquations2D,
                               volume_integral::VolumeIntegralStrongForm,
                               dg::FDSBP, cache)
  D = dg.basis # SBP derivative operator

  dudx = zeros(eltype(u), nnodes(dg), nnodes(dg))
  dudy = zeros(eltype(u), nnodes(dg), nnodes(dg))

  dvdx = zeros(eltype(u), nnodes(dg), nnodes(dg))
  dvdy = zeros(eltype(u), nnodes(dg), nnodes(dg))

  dpdx = zeros(eltype(u), nnodes(dg), nnodes(dg))
  dpdy = zeros(eltype(u), nnodes(dg), nnodes(dg))

  duudx = zeros(eltype(u), nnodes(dg), nnodes(dg))
  dvvdy = zeros(eltype(u), nnodes(dg), nnodes(dg))
  duvdx = zeros(eltype(u), nnodes(dg), nnodes(dg))
  duvdy = zeros(eltype(u), nnodes(dg), nnodes(dg))

  v1v1 = zeros(eltype(u), nnodes(dg), nnodes(dg))
  v2v2 = zeros(eltype(u), nnodes(dg), nnodes(dg))
  v1v2 = zeros(eltype(u), nnodes(dg), nnodes(dg))

  for element in eachelement(dg, cache)
    v1 = view(u, 1, :, :, element)
    v2 = view(u, 2, :, :, element)
    p  = view(u, 3, :, :, element)

    @. v1v1 = v1*v1
    @. v2v2 = v2*v2
    @. v1v2 = v1*v2

    for j in eachnode(dg)
      mul!(view(dudx, :, j) , D, view(v1, :, j))
      mul!(view(dvdx, :, j) , D, view(v2, :, j))
      mul!(view(dpdx, :, j) , D, view(p, :, j) )
      mul!(view(duudx, :, j), D, view(v1v1, :, j))
      mul!(view(duvdx, :, j), D, view(v1v2, :, j))
    end

    for i in eachnode(dg)
      mul!(view(dvdy, i, :) , D, view(v1, i, :))
      mul!(view(dvdy, i, :) , D, view(v2, i, :))
      mul!(view(dpdy, i, :) , D, view(p, i, :) )
      mul!(view(duvdy, i, :), D, view(v1v2, i, :))
      mul!(view(dvvdy, i, :), D, view(v2v2, i, :))
    end

    @. du[1,:,:,element] = 0.5*(v1*dudx + duudx + v2*dudy + duvdy) + dpdx
    @. du[2,:,:,element] = 0.5*(v1*dvdx + duvdx + v2*dvdy + dvvdy) + dpdy
    @. du[3,:,:,element] = dudx + dvdy

  end

  return nothing
end


# Special surface integral for the incompressible Euler equations
#   FIXME: hacked version of the surface integral. Assumes only a single
#          element, so there are no interior surfaces
#   OBS! This is a hacky wall type boundary condition
function calc_surface_integral!(du, u, mesh::TreeMesh{2},
                                equations::IncompressibleEulerEquations2D,
                                surface_integral::SurfaceIntegralStrongForm,
                                dg::FDSBP, cache)
  inv_weight_left  = inv(left_boundary_weight(dg.basis))
  inv_weight_right = inv(right_boundary_weight(dg.basis))

  sat_contribution = zeros(eltype(u), 3)

  @threaded for element in eachelement(dg, cache)
    for l in eachnode(dg)
      # along the bottom
      u_node, v_node, _ = get_node_vars(u, equations, dg, l, 1, element)
      w_n = -v_node
      sat_contribution[1] = -0.5 * inv_weight_left * u_node * w_n
      sat_contribution[2] = -0.5 * inv_weight_left * v_node * w_n
      sat_contribution[3] = -inv_weight_left * w_n
      add_to_node_vars!(du, sat_contribution, equations, dg, l, 1, element)

      # along the right
      u_node, v_node, _ = get_node_vars(u, equations, dg, nnodes(dg), l, element)
      w_n = u_node
      sat_contribution[1] = -0.5 * inv_weight_right * u_node * w_n
      sat_contribution[2] = -0.5 * inv_weight_right * v_node * w_n
      sat_contribution[3] = -inv_weight_right * w_n
      add_to_node_vars!(du, sat_contribution, equations, dg, nnodes(dg), l, element)

      # along the top
      u_node, v_node, _ = get_node_vars(u, equations, dg, l, nnodes(dg), element)
      w_n = v_node
      sat_contribution[1] = -0.5 * inv_weight_right * u_node * w_n
      sat_contribution[2] = -0.5 * inv_weight_right * v_node * w_n
      sat_contribution[3] = -inv_weight_right * w_n
      add_to_node_vars!(du, sat_contribution, equations, dg, l, nnodes(dg), element)

      # along the left
      u_node, v_node, _ = get_node_vars(u, equations, dg, 1, l, element)
      w_n = -u_node
      sat_contribution[1] = -0.5 * inv_weight_left * u_node * w_n
      sat_contribution[2] = -0.5 * inv_weight_left * v_node * w_n
      sat_contribution[3] = -inv_weight_left * w_n
      add_to_node_vars!(du, sat_contribution, equations, dg, 1, l, element)
    end
  end

  return nothing
end
