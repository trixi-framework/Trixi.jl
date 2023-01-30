# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function create_cache_analysis(analyzer, mesh::TreeMesh{3},
                               equations, dg::DG, cache,
                               RealT, uEltype)

  # pre-allocate buffers
  # We use `StrideArray`s here since these buffers are used in performance-critical
  # places and the additional information passed to the compiler makes them faster
  # than native `Array`s.
  u_local = StrideArray(undef, uEltype,
                        StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
  u_tmp1  = StrideArray(undef, uEltype,
                        StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)), StaticInt(nnodes(dg)))
  u_tmp2  = StrideArray(undef, uEltype,
                        StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))
  x_local = StrideArray(undef, RealT,
                        StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
  x_tmp1  = StrideArray(undef, RealT,
                        StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)), StaticInt(nnodes(dg)))
  x_tmp2  = StrideArray(undef, RealT,
                        StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))

  return (; u_local, u_tmp1, u_tmp2, x_local, x_tmp1, x_tmp2)
end


function create_cache_analysis(analyzer, mesh::Union{StructuredMesh{3}, P4estMesh{3}},
                               equations, dg::DG, cache,
                               RealT, uEltype)

  # pre-allocate buffers
  # We use `StrideArray`s here since these buffers are used in performance-critical
  # places and the additional information passed to the compiler makes them faster
  # than native `Array`s.
  u_local = StrideArray(undef, uEltype,
                        StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
  u_tmp1  = StrideArray(undef, uEltype,
                        StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)), StaticInt(nnodes(dg)))
  u_tmp2  = StrideArray(undef, uEltype,
                        StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))
  x_local = StrideArray(undef, RealT,
                        StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
  x_tmp1  = StrideArray(undef, RealT,
                        StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)), StaticInt(nnodes(dg)))
  x_tmp2  = StrideArray(undef, RealT,
                        StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))
  jacobian_local = StrideArray(undef, RealT,
                                StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
  jacobian_tmp1  = StrideArray(undef, RealT,
                               StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)), StaticInt(nnodes(dg)))
  jacobian_tmp2  = StrideArray(undef, RealT,
                               StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))

  return (; u_local, u_tmp1, u_tmp2, x_local, x_tmp1, x_tmp2, jacobian_local, jacobian_tmp1, jacobian_tmp2)
end


function calc_error_norms(func, u, t, analyzer,
                          mesh::TreeMesh{3}, equations, initial_condition,
                          dg::DGSEM, cache, cache_analysis)
  @unpack vandermonde, weights = analyzer
  @unpack node_coordinates = cache.elements
  @unpack u_local, u_tmp1, u_tmp2, x_local, x_tmp1, x_tmp2 = cache_analysis

  # Set up data structures
  l2_error   = zero(func(get_node_vars(u, equations, dg, 1, 1, 1, 1), equations))
  linf_error = copy(l2_error)

  # Iterate over all elements for error calculations
  for element in eachelement(dg, cache)
    # Interpolate solution and node locations to analysis nodes
    multiply_dimensionwise!(u_local, vandermonde, view(u,                :, :, :, :, element), u_tmp1, u_tmp2)
    multiply_dimensionwise!(x_local, vandermonde, view(node_coordinates, :, :, :, :, element), x_tmp1, x_tmp2)

    # Calculate errors at each analysis node
    volume_jacobian_ = volume_jacobian(element, mesh, cache)

    for k in eachnode(analyzer), j in eachnode(analyzer), i in eachnode(analyzer)
      u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j, k), t, equations)
      diff = func(u_exact, equations) - func(get_node_vars(u_local, equations, dg, i, j, k), equations)
      l2_error += diff.^2 * (weights[i] * weights[j] * weights[k] * volume_jacobian_)
      linf_error = @. max(linf_error, abs(diff))
    end
  end

  # For L2 error, divide by total volume
  total_volume_ = total_volume(mesh)
  l2_error = @. sqrt(l2_error / total_volume_)

  return l2_error, linf_error
end


function calc_error_norms(func, u, t, analyzer,
                          mesh::Union{StructuredMesh{3}, P4estMesh{3}},
                          equations, initial_condition,
                          dg::DGSEM, cache, cache_analysis)
  @unpack vandermonde, weights = analyzer
  @unpack node_coordinates, inverse_jacobian = cache.elements
  @unpack u_local, u_tmp1, u_tmp2, x_local, x_tmp1, x_tmp2, jacobian_local, jacobian_tmp1, jacobian_tmp2 = cache_analysis

  # Set up data structures
  l2_error   = zero(func(get_node_vars(u, equations, dg, 1, 1, 1, 1), equations))
  linf_error = copy(l2_error)
  total_volume = zero(real(mesh))

  # Iterate over all elements for error calculations
  for element in eachelement(dg, cache)
    # Interpolate solution and node locations to analysis nodes
    multiply_dimensionwise!(u_local, vandermonde, view(u,                :, :, :, :, element), u_tmp1, u_tmp2)
    multiply_dimensionwise!(x_local, vandermonde, view(node_coordinates, :, :, :, :, element), x_tmp1, x_tmp2)
    multiply_scalar_dimensionwise!(jacobian_local, vandermonde, inv.(view(inverse_jacobian, :, :, :, element)), jacobian_tmp1, jacobian_tmp2)

    # Calculate errors at each analysis node
    @. jacobian_local = abs(jacobian_local)

    for k in eachnode(analyzer), j in eachnode(analyzer), i in eachnode(analyzer)
      u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j, k), t, equations)
      diff = func(u_exact, equations) - func(get_node_vars(u_local, equations, dg, i, j, k), equations)
      l2_error += diff.^2 * (weights[i] * weights[j] * weights[k] * jacobian_local[i, j, k])
      linf_error = @. max(linf_error, abs(diff))
      total_volume += weights[i] * weights[j] * weights[k] * jacobian_local[i, j, k]
    end
  end

  # For L2 error, divide by total volume
  l2_error = @. sqrt(l2_error / total_volume)

  return l2_error, linf_error
end


function integrate_via_indices(func::Func, u,
                               mesh::TreeMesh{3}, equations, dg::DGSEM, cache,
                               args...; normalize=true) where {Func}
  @unpack weights = dg.basis

  # Initialize integral with zeros of the right shape
  integral = zero(func(u, 1, 1, 1, 1, equations, dg, args...))

  # Use quadrature to numerically integrate over entire domain
  for element in eachelement(dg, cache)
    volume_jacobian_ = volume_jacobian(element, mesh, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      integral += volume_jacobian_ * weights[i] * weights[j] * weights[k] * func(u, i, j, k, element, equations, dg, args...)
    end
  end

  # Normalize with total volume
  if normalize
    integral = integral / total_volume(mesh)
  end

  return integral
end


function integrate_via_indices(func::Func, u,
                               mesh::Union{StructuredMesh{3}, P4estMesh{3}},
                               equations, dg::DGSEM, cache,
                               args...; normalize=true) where {Func}
  @unpack weights = dg.basis

  # Initialize integral with zeros of the right shape
  integral = zero(func(u, 1, 1, 1, 1, equations, dg, args...))
  total_volume = zero(real(mesh))

  # Use quadrature to numerically integrate over entire domain
  for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      volume_jacobian = abs(inv(cache.elements.inverse_jacobian[i, j, k, element]))
      integral += volume_jacobian * weights[i] * weights[j] * weights[k] * func(u, i, j, k, element, equations, dg, args...)
      total_volume += volume_jacobian * weights[i] * weights[j] * weights[k]
    end
  end

  # Normalize with total volume
  if normalize
    integral = integral / total_volume
  end

  return integral
end


function integrate(func::Func, u,
                   mesh::Union{TreeMesh{3}, StructuredMesh{3}, P4estMesh{3}},
                   equations, dg::DG, cache; normalize=true) where {Func}
  integrate_via_indices(u, mesh, equations, dg, cache; normalize=normalize) do u, i, j, k, element, equations, dg
    u_local = get_node_vars(u, equations, dg, i, j, k, element)
    return func(u_local, equations)
  end
end


function integrate(func::Func, u,
                   mesh::TreeMesh{3},
                   equations, equations_parabolic,
                   dg::DGSEM,
                   cache, cache_parabolic; normalize=true) where {Func}
  gradients_x, gradients_y, gradients_z = cache_parabolic.gradients
  integrate_via_indices(u, mesh, equations, dg, cache; normalize=normalize) do u, i, j, k, element, equations, dg
    u_local = get_node_vars(u, equations, dg, i, j, k, element)
    gradients_1_local = get_node_vars(gradients_x, equations_parabolic, dg, i, j, k, element)
    gradients_2_local = get_node_vars(gradients_y, equations_parabolic, dg, i, j, k, element)
    gradients_3_local = get_node_vars(gradients_z, equations_parabolic, dg, i, j, k, element)
    return func(u_local, (gradients_1_local, gradients_2_local, gradients_3_local), equations_parabolic)
  end
end


function analyze(::typeof(entropy_timederivative), du, u, t,
                 mesh::Union{TreeMesh{3}, StructuredMesh{3}, P4estMesh{3}},
                 equations, dg::DG, cache)
  # Calculate ∫(∂S/∂u ⋅ ∂u/∂t)dΩ
  integrate_via_indices(u, mesh, equations, dg, cache, du) do u, i, j, k, element, equations, dg, du
    u_node  = get_node_vars(u,  equations, dg, i, j, k, element)
    du_node = get_node_vars(du, equations, dg, i, j, k, element)
    dot(cons2entropy(u_node, equations), du_node)
  end
end



function analyze(::Val{:l2_divb}, du, u, t,
                 mesh::TreeMesh{3}, equations::IdealGlmMhdEquations3D,
                 dg::DGSEM, cache)
  integrate_via_indices(u, mesh, equations, dg, cache, cache, dg.basis.derivative_matrix) do u, i, j, k, element, equations, dg, cache, derivative_matrix
    divb = zero(eltype(u))
    for l in eachnode(dg)
      divb += ( derivative_matrix[i, l] * u[6, l, j, k, element] +
                derivative_matrix[j, l] * u[7, i, l, k, element] +
                derivative_matrix[k, l] * u[8, i, j, l, element] )
    end
    divb *= cache.elements.inverse_jacobian[element]
    divb^2
  end |> sqrt
end

function analyze(::Val{:l2_divb}, du, u, t,
                 mesh::Union{StructuredMesh{3}, P4estMesh{3}}, equations::IdealGlmMhdEquations3D,
                 dg::DGSEM, cache)
  @unpack contravariant_vectors = cache.elements
  integrate_via_indices(u, mesh, equations, dg, cache, cache, dg.basis.derivative_matrix) do u, i, j, k, element, equations, dg, cache, derivative_matrix
    divb = zero(eltype(u))
    # Get the contravariant vectors Ja^1, Ja^2, and Ja^3
    Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    # Compute the transformed divergence
    for l in eachnode(dg)
      divb += ( derivative_matrix[i, l] * (Ja11 * u[6, l, j, k, element] + Ja12 * u[7, l, j, k, element] + Ja13 * u[8, l, j, k, element]) +
                derivative_matrix[j, l] * (Ja21 * u[6, i, l, k, element] + Ja22 * u[7, i, l, k, element] + Ja23 * u[8, i, l, k, element]) +
                derivative_matrix[k, l] * (Ja31 * u[6, i, j, l, element] + Ja32 * u[7, i, j, l, element] + Ja33 * u[8, i, j, l, element]) )
    end
    divb *= cache.elements.inverse_jacobian[i, j, k, element]
    divb^2
  end |> sqrt
end


function analyze(::Val{:linf_divb}, du, u, t,
                 mesh::TreeMesh{3}, equations::IdealGlmMhdEquations3D,
                 dg::DGSEM, cache)
  @unpack derivative_matrix, weights = dg.basis

  # integrate over all elements to get the divergence-free condition errors
  linf_divb = zero(eltype(u))
  for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      divb = zero(eltype(u))
      for l in eachnode(dg)
        divb += ( derivative_matrix[i, l] * u[6, l, j, k, element] +
                  derivative_matrix[j, l] * u[7, i, l, k, element] +
                  derivative_matrix[k, l] * u[8, i, j, l, element] )
      end
      divb *= cache.elements.inverse_jacobian[element]
      linf_divb = max(linf_divb, abs(divb))
    end
  end

  return linf_divb
end

function analyze(::Val{:linf_divb}, du, u, t,
                 mesh::Union{StructuredMesh{3}, P4estMesh{3}}, equations::IdealGlmMhdEquations3D,
                 dg::DGSEM, cache)
  @unpack derivative_matrix, weights = dg.basis
  @unpack contravariant_vectors = cache.elements

  # integrate over all elements to get the divergence-free condition errors
  linf_divb = zero(eltype(u))
  for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      divb = zero(eltype(u))
      # Get the contravariant vectors Ja^1, Ja^2, and Ja^3
      Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
      Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
      Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
      # Compute the transformed divergence
      for l in eachnode(dg)
        divb += ( derivative_matrix[i, l] * (Ja11 * u[6, l, j, k, element] + Ja12 * u[7, l, j, k, element] + Ja13 * u[8, l, j, k, element]) +
                  derivative_matrix[j, l] * (Ja21 * u[6, i, l, k, element] + Ja22 * u[7, i, l, k, element] + Ja23 * u[8, i, l, k, element]) +
                  derivative_matrix[k, l] * (Ja31 * u[6, i, j, l, element] + Ja32 * u[7, i, j, l, element] + Ja33 * u[8, i, j, l, element]) )
      end
      divb *= cache.elements.inverse_jacobian[i, j, k, element]
      linf_divb = max(linf_divb, abs(divb))
    end
  end

  return linf_divb
end


end # @muladd
