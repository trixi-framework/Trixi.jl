# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function calc_acoustic_sources!(acoustic_source_terms, u_euler, u_acoustics, vorticity_mean,
                                coupled_element_ids, mesh,
                                equations::AbstractCompressibleEulerEquations{2}, dg::DGSEM, cache)
  @unpack derivative_matrix = dg.basis

  acoustic_source_terms .= zero(eltype(acoustic_source_terms))

  @threaded for k in 1:length(coupled_element_ids)
    element = coupled_element_ids[k]

    for j in eachnode(dg), i in eachnode(dg)
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

      acoustic_source_terms[1, i, j, k] -= -vorticity_prime * v2_mean -
                                            vorticity_mean[i, j, element] * v2_prime
      acoustic_source_terms[2, i, j, k] -=  vorticity_prime * v1_mean +
                                            vorticity_mean[i, j, element] * v1_prime
    end
  end

  return nothing
end

end # @muladd