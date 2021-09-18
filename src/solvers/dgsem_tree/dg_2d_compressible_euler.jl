# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# Calculate the vorticity on a single node using the derivative matrix from the polynomial basis of
# a DGSEM solver. `u` is the solution on the whole domain.
# This function is used for calculating acoustic source terms for coupled Euler-acoustics
# simulations.
function calc_vorticity_node(u, mesh::TreeMesh{2}, equations::CompressibleEulerEquations2D,
                             dg::DGSEM, cache, i, j, element)
  @unpack derivative_matrix = dg.basis

  v2_x = zero(eltype(u)) # derivative of v2 in x direction
  for ii in eachnode(dg)
    rho, _, rho_v2 = get_node_vars(u, equations, dg, ii, j, element)
    v2 = rho_v2 / rho
    v2_x = v2_x + derivative_matrix[i, ii] * v2
  end

  v1_y = zero(eltype(u)) # derivative of v1 in y direction
  for jj in eachnode(dg)
    rho, rho_v1 = get_node_vars(u, equations, dg, i, jj, element)
    v1 = rho_v1 / rho
    v1_y = v1_y + derivative_matrix[j, jj] * v1
  end

  return (v2_x - v1_y) * cache.elements.inverse_jacobian[element]
end

# Convenience function for calculating the vorticity on the whole domain and storing it in a
# preallocated array
function calc_vorticity!(vorticity, u, mesh::TreeMesh{2}, equations::CompressibleEulerEquations2D,
                         dg::DGSEM, cache)
  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      vorticity[i, j, element] = calc_vorticity_node(u, mesh, equations, dg, cache, i, j, element)
    end
  end

  return nothing
end


end # muladd