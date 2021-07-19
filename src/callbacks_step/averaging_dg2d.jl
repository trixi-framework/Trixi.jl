# Create arrays with DGSEM-specific structure to store the mean values and set them all to 0
function initialize_mean_values(mesh::TreeMesh{2}, equations::AbstractCompressibleEulerEquations{2},
                                dg::DGSEM, cache)
  uEltype = eltype(cache.elements)
  v_mean = zeros(uEltype, (ndims(equations), nnodes(dg), nnodes(dg), nelements(cache.elements)))
  c_mean = zeros(uEltype, (nnodes(dg), nnodes(dg), nelements(cache.elements)))
  rho_mean = zeros(uEltype, size(c_mean))
  vorticity_mean = zeros(uEltype, size(c_mean))

  return (; v_mean, c_mean, rho_mean, vorticity_mean)
end

# Create cache which holds the vorticity for the previous time step. This is needed due to the
# trapezoidal rule
function create_cache(::Type{AveragingCallback}, mesh::TreeMesh{2},
                      equations::AbstractCompressibleEulerEquations{2}, dg::DGSEM, cache)
  # Cache vorticity from previous time step
  uEltype = eltype(cache.elements)
  vorticity_prev = zeros(uEltype, (nnodes(dg), nnodes(dg), nelements(cache.elements)))
  return (; vorticity_prev)
end

# Calculate vorticity for the initial solution and store it in the cache
function initialize_cache!(averaging_callback_cache, u,
                           mesh::TreeMesh{2}, equations::AbstractCompressibleEulerEquations{2},
                           dg::DGSEM, cache)
  @unpack derivative_matrix = dg.basis
  @unpack vorticity_prev = averaging_callback_cache

  # Calculate vorticity for initial solution
  @threaded for element in eachelement(cache.elements)
    for j in eachnode(dg), i in eachnode(dg)
      v2_x = 0.0 # derivative of v2 in x direction
      for ii in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, ii, j, element)
        v2 = u_node[3] / u_node[1]
        v2_x += derivative_matrix[i, ii] * v2
      end

      v1_y = 0.0 # derivative of v1 in y direction
      for jj in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, jj, element)
        v1 = u_node[2] / u_node[1]
        v1_y += derivative_matrix[j, jj] * v1
      end
      vorticity_prev[i, j, element] = (v2_x - v1_y) * cache.elements.inverse_jacobian[element]
    end
  end

  return nothing
end


# Update mean values using the trapezoidal rule
function calc_mean_values!(mean_values, averaging_callback_cache, u, u_prev, integration_constant,
                           mesh::TreeMesh{2}, equations::AbstractCompressibleEulerEquations{2},
                           dg::DGSEM, cache)
  @unpack v_mean, c_mean, rho_mean, vorticity_mean = mean_values
  @unpack vorticity_prev = averaging_callback_cache
  @unpack derivative_matrix = dg.basis

  @threaded for element in eachelement(cache.elements)
    for j in eachnode(dg), i in eachnode(dg)
      # Calculate vorticity
      v2_x = 0.0 # derivative of v2 in x direction
      for ii in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, ii, j, element)
        v2 = u_node[3] / u_node[1]
        v2_x += derivative_matrix[i, ii] * v2
      end

      v1_y = 0.0 # derivative of v1 in y direction
      for jj in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, jj, element)
        v1 = u_node[2] / u_node[1]
        v1_y += derivative_matrix[j, jj] * v1
      end
      vorticity = (v2_x - v1_y) * cache.elements.inverse_jacobian[element]
      vorticity_prev_node = vorticity_prev[i, j, element]
      vorticity_prev[i, j, element] = vorticity # Cache current velocity for the next time step


      u_node_prim = cons2prim(get_node_vars(u, equations, dg, i, j, element), equations)
      u_prev_node_prim = cons2prim(get_node_vars(u_prev, equations, dg, i, j, element), equations)

      rho = u_node_prim[1]
      v1 = u_node_prim[2]
      v2 = u_node_prim[3]
      p = u_node_prim[4]

      rho_prev = u_prev_node_prim[1]
      v1_prev = u_prev_node_prim[2]
      v2_prev = u_prev_node_prim[3]
      p_prev = u_prev_node_prim[4]

      c = sqrt(equations.gamma * p / rho)
      c_prev = sqrt(equations.gamma * p_prev / rho_prev)

      # Calculate the contribution to the mean values using the trapezoidal rule
      vorticity_mean[i, j, element] += integration_constant * (vorticity_prev_node + vorticity)
      v_mean[1, i, j, element]      += integration_constant * (v1_prev + v1)
      v_mean[2, i, j, element]      += integration_constant * (v2_prev + v2)
      c_mean[i, j, element]         += integration_constant * (c_prev + c)
      rho_mean[i, j, element]       += integration_constant * (rho_prev + rho)
    end
  end

  return nothing
end