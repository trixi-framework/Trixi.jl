function calc_gradient_c_mean_square!(grad_c_mean_sq, u, mesh, equations::AcousticPerturbationEquations2D,
                                      dg::DGSEM, cache)
  @unpack derivative_matrix = dg.basis

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      c_mean_sq_x = 0.0 # partial derivative of c_mean square in x direction on the ref. element
      for ii in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, ii, j, element)
        c_mean = u_node[6]
        c_mean_sq_x += derivative_matrix[i, ii] * c_mean^2
      end
      grad_c_mean_sq[1, i, j, element] = c_mean_sq_x * cache.elements.inverse_jacobian[element]

      c_mean_sq_y = 0.0 # partial derivative of c_mean square in y direction on the ref. element
      for jj in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, jj, element)
        c_mean = u_node[6]
        c_mean_sq_y += derivative_matrix[j, jj] * c_mean^2
      end
      grad_c_mean_sq[2, i, j, element] = c_mean_sq_y * cache.elements.inverse_jacobian[element]
    end
  end

  return nothing
end


function calc_acoustic_sources!(acoustic_source_terms, u_euler, u_ape, vorticity_mean,
                                source_region, weights, mesh, equations, dg::DGSEM, cache)
  @unpack derivative_matrix = dg.basis
  @unpack node_coordinates = cache.elements

  acoustic_source_terms .= zero(eltype(acoustic_source_terms))

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      # Only calculate sources on nodes that lie within the acoustic source region
      if source_region(x)
        # Calculate vorticity
        dv2_dx = 0.0
        for ii in eachnode(dg)
          # TODO: Use get_node_vars instead?
          v2 = u_euler[3, ii, j, element] / u_euler[1, ii, j, element]
          dv2_dx += derivative_matrix[i, ii] * v2
        end

        dv1_dy = 0.0
        for jj in eachnode(dg)
          v1 = u_euler[2, i, jj, element] / u_euler[1, i, jj, element]
          dv1_dy += derivative_matrix[j, jj] * v1
        end

        vorticity = (dv2_dx - dv1_dy) * cache.elements.inverse_jacobian[element]

        prim_euler = cons2prim(get_node_vars(u_euler, equations, dg, i, j, element), equations)
        v1 = prim_euler[2]
        v2 = prim_euler[3]
        v1_mean = u_ape[4, i, j, element]
        v2_mean = u_ape[5, i, j, element]

        vorticity_prime = vorticity - vorticity_mean[i, j, element]
        v1_prime = v1 - v1_mean
        v2_prime = v2 - v2_mean

        acoustic_source_terms[1, i, j, element] -= -vorticity_prime * v2_mean - vorticity_mean[i, j, element] * v2_prime
        acoustic_source_terms[2, i, j, element] -=  vorticity_prime * v1_mean + vorticity_mean[i, j, element] * v1_prime

        # Apply acoustic source weighting function
        acoustic_source_terms[1, i, j, element] *= weights(x)
        acoustic_source_terms[2, i, j, element] *= weights(x)
      end
    end
  end

  return nothing
end