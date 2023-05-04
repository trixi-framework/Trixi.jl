# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function create_cache_analysis(analyzer, mesh::TreeMesh{2},
                               equations, dg::DG, cache,
                               RealT, uEltype)

  # pre-allocate buffers
  # We use `StrideArray`s here since these buffers are used in performance-critical
  # places and the additional information passed to the compiler makes them faster
  # than native `Array`s.
  u_local = StrideArray(undef, uEltype,
                        StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
  u_tmp1  = StrideArray(undef, uEltype,
                        StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))
  x_local = StrideArray(undef, RealT,
                        StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
  x_tmp1  = StrideArray(undef, RealT,
                        StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))

  return (; u_local, u_tmp1, x_local, x_tmp1)
end


function create_cache_analysis(analyzer, mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                               equations, dg::DG, cache,
                               RealT, uEltype)

  # pre-allocate buffers
  # We use `StrideArray`s here since these buffers are used in performance-critical
  # places and the additional information passed to the compiler makes them faster
  # than native `Array`s.
  u_local = StrideArray(undef, uEltype,
                        StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
  u_tmp1  = StrideArray(undef, uEltype,
                        StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))
  x_local = StrideArray(undef, RealT,
                        StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
  x_tmp1  = StrideArray(undef, RealT,
                        StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))
  jacobian_local = StrideArray(undef, RealT,
                               StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
  jacobian_tmp1  = StrideArray(undef, RealT,
                               StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))

  return (; u_local, u_tmp1, x_local, x_tmp1, jacobian_local, jacobian_tmp1)
end


function calc_error_norms(func, u, t, analyzer,
                          mesh::TreeMesh{2}, equations, initial_condition,
                          dg::DGSEM, cache, cache_analysis)
  @unpack vandermonde, weights = analyzer
  @unpack node_coordinates = cache.elements
  @unpack u_local, u_tmp1, x_local, x_tmp1 = cache_analysis

  # Set up data structures
  l2_error   = zero(func(get_node_vars(u, equations, dg, 1, 1, 1), equations))
  linf_error = copy(l2_error)

  # Iterate over all elements for error calculations
  # Accumulate L2 error on the element first so that the order of summation is the
  # same as in the parallel case to ensure exact equality. This facilitates easier parallel
  # development and debugging (see
  # https://github.com/trixi-framework/Trixi.jl/pull/850#pullrequestreview-757463943 for details).
  for element in eachelement(dg, cache)
    # Set up data structures for local element L2 error
    l2_error_local = zero(l2_error)

    # Interpolate solution and node locations to analysis nodes
    multiply_dimensionwise!(u_local, vandermonde, view(u,                :, :, :, element), u_tmp1)
    multiply_dimensionwise!(x_local, vandermonde, view(node_coordinates, :, :, :, element), x_tmp1)

    # Calculate errors at each analysis node
    volume_jacobian_ = volume_jacobian(element, mesh, cache)

    for j in eachnode(analyzer), i in eachnode(analyzer)
      u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j), t, equations)
      diff = func(u_exact, equations) - func(get_node_vars(u_local, equations, dg, i, j), equations)
      l2_error_local += diff.^2 * (weights[i] * weights[j] * volume_jacobian_)
      linf_error = @. max(linf_error, abs(diff))
    end
    l2_error += l2_error_local
  end

  # For L2 error, divide by total volume
  total_volume_ = total_volume(mesh)
  l2_error = @. sqrt(l2_error / total_volume_)

  return l2_error, linf_error
end


function calc_error_norms(func, u, t, analyzer,
                          mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}}, equations,
                          initial_condition, dg::DGSEM, cache, cache_analysis)
  @unpack vandermonde, weights = analyzer
  @unpack node_coordinates, inverse_jacobian = cache.elements
  @unpack u_local, u_tmp1, x_local, x_tmp1, jacobian_local, jacobian_tmp1 = cache_analysis

  # Set up data structures
  l2_error   = zero(func(get_node_vars(u, equations, dg, 1, 1, 1), equations))
  linf_error = copy(l2_error)
  total_volume = zero(real(mesh))

  # Iterate over all elements for error calculations
  for element in eachelement(dg, cache)
    # Interpolate solution and node locations to analysis nodes
    multiply_dimensionwise!(u_local, vandermonde, view(u,                :, :, :, element), u_tmp1)
    multiply_dimensionwise!(x_local, vandermonde, view(node_coordinates, :, :, :, element), x_tmp1)
    multiply_scalar_dimensionwise!(jacobian_local, vandermonde, inv.(view(inverse_jacobian, :, :, element)), jacobian_tmp1)

    # Calculate errors at each analysis node
    @. jacobian_local = abs(jacobian_local)

    for j in eachnode(analyzer), i in eachnode(analyzer)
      u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j), t, equations)
      diff = func(u_exact, equations) - func(get_node_vars(u_local, equations, dg, i, j), equations)
      l2_error += diff.^2 * (weights[i] * weights[j] * jacobian_local[i, j])
      linf_error = @. max(linf_error, abs(diff))
      total_volume += weights[i] * weights[j] * jacobian_local[i, j]
    end
  end

  # For L2 error, divide by total volume
  l2_error = @. sqrt(l2_error / total_volume)

  return l2_error, linf_error
end


function integrate_via_indices(func::Func, u,
                               mesh::TreeMesh{2}, equations, dg::DGSEM, cache,
                               args...; normalize=true) where {Func}
  @unpack weights = dg.basis

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


function integrate_via_indices(func::Func, u,
                               mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}}, equations,
                               dg::DGSEM, cache, args...; normalize=true) where {Func}
  @unpack weights = dg.basis

  # Initialize integral with zeros of the right shape
  integral = zero(func(u, 1, 1, 1, equations, dg, args...))
  total_volume = zero(real(mesh))

  # Use quadrature to numerically integrate over entire domain
  for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      volume_jacobian = abs(inv(cache.elements.inverse_jacobian[i, j, element]))
      integral += volume_jacobian * weights[i] * weights[j] * func(u, i, j, element, equations, dg, args...)
      total_volume += volume_jacobian * weights[i] * weights[j]
    end
  end

  # Normalize with total volume
  if normalize
    integral = integral / total_volume
  end

  return integral
end


function integrate(func::Func, u,
                   mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                   equations, dg::DG, cache; normalize=true) where {Func}
  integrate_via_indices(u, mesh, equations, dg, cache; normalize=normalize) do u, i, j, element, equations, dg
    u_local = get_node_vars(u, equations, dg, i, j, element)
    return func(u_local, equations)
  end
end

function analyze(::typeof(entropy_timederivative), du, u, t,
                 mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                 equations, dg::DG, cache)
  # Calculate ∫(∂S/∂u ⋅ ∂u/∂t)dΩ
  integrate_via_indices(u, mesh, equations, dg, cache, du) do u, i, j, element, equations, dg, du
    u_node  = get_node_vars(u,  equations, dg, i, j, element)
    du_node = get_node_vars(du, equations, dg, i, j, element)
    dot(cons2entropy(u_node, equations), du_node)
  end
end



function analyze(::Val{:l2_divb}, du, u, t,
                 mesh::TreeMesh{2},
                 equations::IdealGlmMhdEquations2D, dg::DGSEM, cache)
  integrate_via_indices(u, mesh, equations, dg, cache, cache, dg.basis.derivative_matrix) do u, i, j, element, equations, dg, cache, derivative_matrix
    divb = zero(eltype(u))
    for k in eachnode(dg)
      divb += ( derivative_matrix[i, k] * u[6, k, j, element] +
                derivative_matrix[j, k] * u[7, i, k, element] )
    end
    divb *= cache.elements.inverse_jacobian[element]
    divb^2
  end |> sqrt
end

function analyze(::Val{:l2_divb}, du, u, t,
                 mesh::TreeMesh{2}, equations::IdealGlmMhdMulticomponentEquations2D,
                 dg::DG, cache)
  integrate_via_indices(u, mesh, equations, dg, cache, cache, dg.basis.derivative_matrix) do u, i, j, element, equations, dg, cache, derivative_matrix
    divb = zero(eltype(u))
    for k in eachnode(dg)
      divb += ( derivative_matrix[i, k] * u[5, k, j, element] +
                derivative_matrix[j, k] * u[6, i, k, element] )
    end
    divb *= cache.elements.inverse_jacobian[element]
    divb^2
  end |> sqrt
end

function analyze(::Val{:l2_divb}, du, u, t,
                 mesh::Union{StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
                 equations::IdealGlmMhdEquations2D, dg::DGSEM, cache)
  @unpack contravariant_vectors = cache.elements
  integrate_via_indices(u, mesh, equations, dg, cache, cache, dg.basis.derivative_matrix) do u, i, j, element, equations, dg, cache, derivative_matrix
    divb = zero(eltype(u))
    # Get the contravariant vectors Ja^1 and Ja^2
    Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
    Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
    # Compute the transformed divergence
    for k in eachnode(dg)
      divb += ( derivative_matrix[i, k] * (Ja11 * u[6, k, j, element] + Ja12 * u[7, k, j, element]) +
                derivative_matrix[j, k] * (Ja21 * u[6, i, k, element] + Ja22 * u[7, i, k, element]) )
    end
    divb *= cache.elements.inverse_jacobian[i, j, element]
    divb^2
  end |> sqrt
end


function analyze(::Val{:linf_divb}, du, u, t,
                 mesh::TreeMesh{2},
                 equations::IdealGlmMhdEquations2D, dg::DGSEM, cache)
  @unpack derivative_matrix, weights = dg.basis

  # integrate over all elements to get the divergence-free condition errors
  linf_divb = zero(eltype(u))
  for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      divb = zero(eltype(u))
      for k in eachnode(dg)
        divb += ( derivative_matrix[i, k] * u[6, k, j, element] +
                  derivative_matrix[j, k] * u[7, i, k, element] )
      end
      divb *= cache.elements.inverse_jacobian[element]
      linf_divb = max(linf_divb, abs(divb))
    end
  end

  return linf_divb
end

function analyze(::Val{:linf_divb}, du, u, t,
                 mesh::TreeMesh{2}, equations::IdealGlmMhdMulticomponentEquations2D,
                 dg::DG, cache)
  @unpack derivative_matrix, weights = dg.basis

  # integrate over all elements to get the divergence-free condition errors
  linf_divb = zero(eltype(u))
  for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      divb = zero(eltype(u))
      for k in eachnode(dg)
        divb += ( derivative_matrix[i, k] * u[5, k, j, element] +
                  derivative_matrix[j, k] * u[6, i, k, element] )
      end
      divb *= cache.elements.inverse_jacobian[element]
      linf_divb = max(linf_divb, abs(divb))
    end
  end

  return linf_divb
end

function analyze(::Val{:linf_divb}, du, u, t,
                 mesh::Union{StructuredMesh{2},UnstructuredMesh2D,P4estMesh{2}},
                 equations::IdealGlmMhdEquations2D, dg::DGSEM, cache)
  @unpack derivative_matrix, weights = dg.basis
  @unpack contravariant_vectors = cache.elements

  # integrate over all elements to get the divergence-free condition errors
  linf_divb = zero(eltype(u))
  for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      divb = zero(eltype(u))
      # Get the contravariant vectors Ja^1 and Ja^2
      Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
      Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
      # Compute the transformed divergence
      for k in eachnode(dg)
        divb += ( derivative_matrix[i, k] * (Ja11 * u[6, k, j, element] + Ja12 * u[7, k, j, element]) +
                  derivative_matrix[j, k] * (Ja21 * u[6, i, k, element] + Ja22 * u[7, i, k, element]) )
      end
      divb *= cache.elements.inverse_jacobian[i, j, element]
      linf_divb = max(linf_divb, abs(divb))
    end
  end

  return linf_divb
end


end # @muladd
