# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function calc_gradient_c_mean_square!(grad_c_mean_sq, u, mesh,
                                      equations::AcousticPerturbationEquations2D, dg::DGSEM, cache)
  @unpack derivative_matrix = dg.basis

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      c_mean_sq_x = zero(eltype(u)) # partial derivative of c_mean square in x direction on the ref. element
      for ii in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, ii, j, element)
        c_mean = u_node[6]
        c_mean_sq_x += derivative_matrix[i, ii] * c_mean^2
      end
      grad_c_mean_sq[1, i, j, element] = c_mean_sq_x * cache.elements.inverse_jacobian[element]

      c_mean_sq_y = zero(eltype(u)) # partial derivative of c_mean square in y direction on the ref. element
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


function calc_acoustic_sources!(acoustic_source_terms, u_euler, u_acoustics, vorticity_mean,
                                source_region, weights, mesh,
                                equations::AbstractCompressibleEulerEquations{2}, dg::DGSEM, cache)
  @unpack derivative_matrix = dg.basis
  @unpack node_coordinates = cache.elements

  acoustic_source_terms .= zero(eltype(acoustic_source_terms))

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      # Only calculate sources on nodes that lie within the acoustic source region
      if source_region(x)
        # Calculate vorticity
        v2_x = zero(eltype(u_euler)) # derivative of v2 in x direction
        for ii in eachnode(dg)
          u_euler_node = get_node_vars(u_euler, equations, dg, ii, j, element)
          v2 = u_euler_node[3] / u_euler_node[1]
          v2_x += derivative_matrix[i, ii] * v2
        end

        v1_y = zero(eltype(u_euler)) # derivative of v1 in y direction
        for jj in eachnode(dg)
          u_euler_node = get_node_vars(u_euler, equations, dg, i, jj, element)
          v1 = u_euler_node[2] / u_euler_node[1]
          v1_y += derivative_matrix[j, jj] * v1
        end

        vorticity = (v2_x - v1_y) * cache.elements.inverse_jacobian[element]

        prim_euler = cons2prim(get_node_vars(u_euler, equations, dg, i, j, element), equations)
        v1 = prim_euler[2]
        v2 = prim_euler[3]
        v1_mean = u_acoustics[4, i, j, element]
        v2_mean = u_acoustics[5, i, j, element]

        vorticity_prime = vorticity - vorticity_mean[i, j, element]
        v1_prime = v1 - v1_mean
        v2_prime = v2 - v2_mean

        acoustic_source_terms[1, i, j, element] -= -vorticity_prime * v2_mean -
                                                    vorticity_mean[i, j, element] * v2_prime
        acoustic_source_terms[2, i, j, element] -=  vorticity_prime * v1_mean +
                                                    vorticity_mean[i, j, element] * v1_prime

        # Apply acoustic source weighting function
        acoustic_source_terms[1, i, j, element] *= weights(x)
        acoustic_source_terms[2, i, j, element] *= weights(x)
      end
    end
  end

  return nothing
end

end # @muladd