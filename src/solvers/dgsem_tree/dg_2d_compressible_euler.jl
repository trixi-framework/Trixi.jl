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



# From here on, this file contains specializations of DG methods on the
# TreeMesh2D to the compressible Euler equations.
#
# The specialized methods contain relatively verbose and ugly code in the sense
# that we do not use the common "pointwise physics" interface of Trixi.jl.
# However, this is currently (November 2021) necessary to get a significant
# speed-up by using SIMD instructions via LoopVectorization.jl.
#
# TODO: SIMD. We could get even more speed-up if we did not need to permute
#             array dimensions, i.e., if we changed the basic memory layout.
#
# We do not wrap this code in `@muladd begin ... end` block. Optimizations like
# this are handled automatically by LoopVectorization.jl.


# We specialize on `PtrArray` since these will be returned by `Trixi.wrap_array`
# if LoopVectorization.jl can handle the array types. This ensures that `@turbo`
# works efficiently here.
@inline function split_form_kernel!(_du::PtrArray, u_cons::PtrArray,
                                    element, mesh::TreeMesh{2},
                                    nonconservative_terms::Val{false},
                                    equations::CompressibleEulerEquations2D,
                                    volume_flux::typeof(flux_shima_etal),
                                    dg::DGSEM, cache, alpha)
  @unpack derivative_split = dg.basis

  # Create a temporary array that will be used to store the RHS with permuted
  # indices `[i, j, v]` to allow using SIMD instructions.
  # `StrideArray`s with purely static dimensions do not allocate on the heap.
  du = StrideArray{eltype(u_cons)}(undef,
    (ntuple(_ -> StaticInt(nnodes(dg)), ndims(mesh))...,
     StaticInt(nvariables(equations))))

  # Convert conserved to primitive variables on the given `element`.
  u_prim = StrideArray{eltype(u_cons)}(undef,
    (ntuple(_ -> StaticInt(nnodes(dg)), ndims(mesh))...,
     StaticInt(nvariables(equations))))

  @turbo for j in eachnode(dg), i in eachnode(dg)
    rho    = u_cons[1, i, j, element]
    rho_v1 = u_cons[2, i, j, element]
    rho_v2 = u_cons[3, i, j, element]
    rho_e  = u_cons[4, i, j, element]

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))

    u_prim[i, j, 1] = rho
    u_prim[i, j, 2] = v1
    u_prim[i, j, 3] = v2
    u_prim[i, j, 4] = p
  end


  # x direction
  # At first, we create new temporary arrays with permuted memory layout to
  # allow using SIMD instructions along the first dimension (which is contiguous
  # in memory).
  du_permuted = StrideArray{eltype(u_cons)}(undef,
    (StaticInt(nnodes(dg)), StaticInt(nnodes(dg)),
     StaticInt(nvariables(equations))))

  u_prim_permuted = StrideArray{eltype(u_cons)}(undef,
    (StaticInt(nnodes(dg)), StaticInt(nnodes(dg)),
     StaticInt(nvariables(equations))))

  @turbo for v in eachvariable(equations),
             j in eachnode(dg),
             i in eachnode(dg)
    u_prim_permuted[j, i, v] = u_prim[i, j, v]
  end
  fill!(du_permuted, zero(eltype(du_permuted)))

  # Next, we basically inline the volume flux. To allow SIMD vectorization and
  # still use the symmetry of the volume flux and the derivative matrix, we
  # loop over the triangular part in an outer loop and use a plain inner loop.
  for i in eachnode(dg), ii in (i+1):nnodes(dg)
    @turbo for j in eachnode(dg)
      rho_ll = u_prim_permuted[j, i, 1]
      v1_ll  = u_prim_permuted[j, i, 2]
      v2_ll  = u_prim_permuted[j, i, 3]
      p_ll   = u_prim_permuted[j, i, 4]

      rho_rr = u_prim_permuted[j, ii, 1]
      v1_rr  = u_prim_permuted[j, ii, 2]
      v2_rr  = u_prim_permuted[j, ii, 3]
      p_rr   = u_prim_permuted[j, ii, 4]

      # Compute required mean values
      rho_avg = 0.5 * (rho_ll + rho_rr)
      v1_avg  = 0.5 * ( v1_ll +  v1_rr)
      v2_avg  = 0.5 * ( v2_ll +  v2_rr)
      p_avg   = 0.5 * (  p_ll +   p_rr)
      kin_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr)
      pv1_avg = 0.5 * (p_ll * v1_rr + p_rr * v1_ll)

      # Calculate fluxes depending on Cartesian orientation
      f1 = rho_avg * v1_avg
      f2 = f1 * v1_avg + p_avg
      f3 = f1 * v2_avg
      f4 = p_avg * v1_avg * equations.inv_gamma_minus_one + f1 * kin_avg + pv1_avg

      # Add scaled fluxes to RHS
      factor_i = alpha * derivative_split[i, ii]
      du_permuted[j, i, 1] += factor_i * f1
      du_permuted[j, i, 2] += factor_i * f2
      du_permuted[j, i, 3] += factor_i * f3
      du_permuted[j, i, 4] += factor_i * f4

      factor_ii = alpha * derivative_split[ii, i]
      du_permuted[j, ii, 1] += factor_ii * f1
      du_permuted[j, ii, 2] += factor_ii * f2
      du_permuted[j, ii, 3] += factor_ii * f3
      du_permuted[j, ii, 4] += factor_ii * f4
    end
  end

  @turbo for v in eachvariable(equations),
             j in eachnode(dg),
             i in eachnode(dg)
    du[i, j, v] = du_permuted[j, i, v]
  end


  # y direction
  # The memory layout is already optimal for SIMD vectorization in this loop.
  for j in eachnode(dg), jj in (j+1):nnodes(dg)
    @turbo for i in eachnode(dg)
      rho_ll = u_prim[i, j, 1]
      v1_ll  = u_prim[i, j, 2]
      v2_ll  = u_prim[i, j, 3]
      p_ll   = u_prim[i, j, 4]

      rho_rr = u_prim[i, jj, 1]
      v1_rr  = u_prim[i, jj, 2]
      v2_rr  = u_prim[i, jj, 3]
      p_rr   = u_prim[i, jj, 4]

      # Compute required mean values
      rho_avg = 0.5 * (rho_ll + rho_rr)
      v1_avg  = 0.5 * ( v1_ll +  v1_rr)
      v2_avg  = 0.5 * ( v2_ll +  v2_rr)
      p_avg   = 0.5 * (  p_ll +   p_rr)
      kin_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr)
      pv2_avg = 0.5 * (p_ll * v2_rr + p_rr * v2_ll)

      # Calculate fluxes depending on Cartesian orientation
      f1 = rho_avg * v2_avg
      f2 = f1 * v1_avg
      f3 = f1 * v2_avg + p_avg
      f4 = p_avg*v2_avg * equations.inv_gamma_minus_one + f1 * kin_avg + pv2_avg

      # Add scaled fluxes to RHS
      factor_j = alpha * derivative_split[j, jj]
      du[i, j, 1] += factor_j * f1
      du[i, j, 2] += factor_j * f2
      du[i, j, 3] += factor_j * f3
      du[i, j, 4] += factor_j * f4

      factor_jj = alpha * derivative_split[jj, j]
      du[i, jj, 1] += factor_jj * f1
      du[i, jj, 2] += factor_jj * f2
      du[i, jj, 3] += factor_jj * f3
      du[i, jj, 4] += factor_jj * f4
    end
  end


  # Finally, we add the temporary RHS computed here to the global RHS in the
  # given `element`.
  @turbo for v in eachvariable(equations),
             j in eachnode(dg),
             i in eachnode(dg)
    _du[v, i, j, element] += du[i, j, v]
  end
end
