
# From here on, this file contains specializations of DG methods on the
# TreeMesh3D to the compressible Euler equations.
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
@inline function flux_differencing_kernel!(_du::PtrArray, u_cons::PtrArray,
                                           element, mesh::TreeMesh{3},
                                           nonconservative_terms::False,
                                           equations::CompressibleEulerEquations3D,
                                           volume_flux::typeof(flux_shima_etal_turbo),
                                           dg::DGSEM, cache, alpha)
    @unpack derivative_split = dg.basis

    # Create a temporary array that will be used to store the RHS with permuted
    # indices `[i, j, k, v]` to allow using SIMD instructions.
    # `StrideArray`s with purely static dimensions do not allocate on the heap.
    du = StrideArray{eltype(u_cons)}(undef,
                                     (ntuple(_ -> StaticInt(nnodes(dg)), ndims(mesh))...,
                                      StaticInt(nvariables(equations))))

    # Convert conserved to primitive variables on the given `element`.
    u_prim = StrideArray{eltype(u_cons)}(undef,
                                         (ntuple(_ -> StaticInt(nnodes(dg)),
                                                 ndims(mesh))...,
                                          StaticInt(nvariables(equations))))

    @turbo for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        rho = u_cons[1, i, j, k, element]
        rho_v1 = u_cons[2, i, j, k, element]
        rho_v2 = u_cons[3, i, j, k, element]
        rho_v3 = u_cons[4, i, j, k, element]
        rho_e = u_cons[5, i, j, k, element]

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        p = (equations.gamma - 1) *
            (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3))

        u_prim[i, j, k, 1] = rho
        u_prim[i, j, k, 2] = v1
        u_prim[i, j, k, 3] = v2
        u_prim[i, j, k, 4] = v3
        u_prim[i, j, k, 5] = p
    end

    # x direction
    # At first, we create new temporary arrays with permuted memory layout to
    # allow using SIMD instructions along the first dimension (which is contiguous
    # in memory).
    du_permuted = StrideArray{eltype(u_cons)}(undef,
                                              (StaticInt(nnodes(dg)^2),
                                               StaticInt(nnodes(dg)),
                                               StaticInt(nvariables(equations))))

    u_prim_permuted = StrideArray{eltype(u_cons)}(undef,
                                                  (StaticInt(nnodes(dg)^2),
                                                   StaticInt(nnodes(dg)),
                                                   StaticInt(nvariables(equations))))

    @turbo for v in eachvariable(equations),
               k in eachnode(dg),
               j in eachnode(dg),
               i in eachnode(dg)

        jk = j + nnodes(dg) * (k - 1)
        u_prim_permuted[jk, i, v] = u_prim[i, j, k, v]
    end
    fill!(du_permuted, zero(eltype(du_permuted)))

    # Next, we basically inline the volume flux. To allow SIMD vectorization and
    # still use the symmetry of the volume flux and the derivative matrix, we
    # loop over the triangular part in an outer loop and use a plain inner loop.
    for i in eachnode(dg), ii in (i + 1):nnodes(dg)
        @turbo for jk in Base.OneTo(nnodes(dg)^2)
            rho_ll = u_prim_permuted[jk, i, 1]
            v1_ll = u_prim_permuted[jk, i, 2]
            v2_ll = u_prim_permuted[jk, i, 3]
            v3_ll = u_prim_permuted[jk, i, 4]
            p_ll = u_prim_permuted[jk, i, 5]

            rho_rr = u_prim_permuted[jk, ii, 1]
            v1_rr = u_prim_permuted[jk, ii, 2]
            v2_rr = u_prim_permuted[jk, ii, 3]
            v3_rr = u_prim_permuted[jk, ii, 4]
            p_rr = u_prim_permuted[jk, ii, 5]

            # Compute required mean values
            rho_avg = 0.5 * (rho_ll + rho_rr)
            v1_avg = 0.5 * (v1_ll + v1_rr)
            v2_avg = 0.5 * (v2_ll + v2_rr)
            v3_avg = 0.5 * (v3_ll + v3_rr)
            p_avg = 0.5 * (p_ll + p_rr)
            kin_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)
            pv1_avg = 0.5 * (p_ll * v1_rr + p_rr * v1_ll)

            # Calculate fluxes depending on Cartesian orientation
            f1 = rho_avg * v1_avg
            f2 = f1 * v1_avg + p_avg
            f3 = f1 * v2_avg
            f4 = f1 * v3_avg
            f5 = p_avg * v1_avg * equations.inv_gamma_minus_one + f1 * kin_avg + pv1_avg

            # Add scaled fluxes to RHS
            factor_i = alpha * derivative_split[i, ii]
            du_permuted[jk, i, 1] += factor_i * f1
            du_permuted[jk, i, 2] += factor_i * f2
            du_permuted[jk, i, 3] += factor_i * f3
            du_permuted[jk, i, 4] += factor_i * f4
            du_permuted[jk, i, 5] += factor_i * f5

            factor_ii = alpha * derivative_split[ii, i]
            du_permuted[jk, ii, 1] += factor_ii * f1
            du_permuted[jk, ii, 2] += factor_ii * f2
            du_permuted[jk, ii, 3] += factor_ii * f3
            du_permuted[jk, ii, 4] += factor_ii * f4
            du_permuted[jk, ii, 5] += factor_ii * f5
        end
    end

    @turbo for v in eachvariable(equations),
               k in eachnode(dg),
               j in eachnode(dg),
               i in eachnode(dg)

        jk = j + nnodes(dg) * (k - 1)
        du[i, j, k, v] = du_permuted[jk, i, v]
    end

    # y direction
    # A possible permutation of array dimensions with improved opportunities for
    # SIMD vectorization appeared to be slower than the direct version used here
    # in preliminary numerical experiments on an AVX2 system.
    for j in eachnode(dg), jj in (j + 1):nnodes(dg)
        @turbo for k in eachnode(dg), i in eachnode(dg)
            rho_ll = u_prim[i, j, k, 1]
            v1_ll = u_prim[i, j, k, 2]
            v2_ll = u_prim[i, j, k, 3]
            v3_ll = u_prim[i, j, k, 4]
            p_ll = u_prim[i, j, k, 5]

            rho_rr = u_prim[i, jj, k, 1]
            v1_rr = u_prim[i, jj, k, 2]
            v2_rr = u_prim[i, jj, k, 3]
            v3_rr = u_prim[i, jj, k, 4]
            p_rr = u_prim[i, jj, k, 5]

            # Compute required mean values
            rho_avg = 0.5 * (rho_ll + rho_rr)
            v1_avg = 0.5 * (v1_ll + v1_rr)
            v2_avg = 0.5 * (v2_ll + v2_rr)
            v3_avg = 0.5 * (v3_ll + v3_rr)
            p_avg = 0.5 * (p_ll + p_rr)
            kin_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)
            pv2_avg = 0.5 * (p_ll * v2_rr + p_rr * v2_ll)

            # Calculate fluxes depending on Cartesian orientation
            f1 = rho_avg * v2_avg
            f2 = f1 * v1_avg
            f3 = f1 * v2_avg + p_avg
            f4 = f1 * v3_avg
            f5 = p_avg * v2_avg * equations.inv_gamma_minus_one + f1 * kin_avg + pv2_avg

            # Add scaled fluxes to RHS
            factor_j = alpha * derivative_split[j, jj]
            du[i, j, k, 1] += factor_j * f1
            du[i, j, k, 2] += factor_j * f2
            du[i, j, k, 3] += factor_j * f3
            du[i, j, k, 4] += factor_j * f4
            du[i, j, k, 5] += factor_j * f5

            factor_jj = alpha * derivative_split[jj, j]
            du[i, jj, k, 1] += factor_jj * f1
            du[i, jj, k, 2] += factor_jj * f2
            du[i, jj, k, 3] += factor_jj * f3
            du[i, jj, k, 4] += factor_jj * f4
            du[i, jj, k, 5] += factor_jj * f5
        end
    end

    # z direction
    # The memory layout is already optimal for SIMD vectorization in this loop.
    # We just squeeze the first two dimensions to make the code slightly faster.
    GC.@preserve u_prim begin
        u_prim_reshaped = PtrArray(pointer(u_prim),
                                   (StaticInt(nnodes(dg)^2), StaticInt(nnodes(dg)),
                                    StaticInt(nvariables(equations))))

        du_reshaped = PtrArray(pointer(du),
                               (StaticInt(nnodes(dg)^2), StaticInt(nnodes(dg)),
                                StaticInt(nvariables(equations))))

        for k in eachnode(dg), kk in (k + 1):nnodes(dg)
            @turbo for ij in Base.OneTo(nnodes(dg)^2)
                rho_ll = u_prim_reshaped[ij, k, 1]
                v1_ll = u_prim_reshaped[ij, k, 2]
                v2_ll = u_prim_reshaped[ij, k, 3]
                v3_ll = u_prim_reshaped[ij, k, 4]
                p_ll = u_prim_reshaped[ij, k, 5]

                rho_rr = u_prim_reshaped[ij, kk, 1]
                v1_rr = u_prim_reshaped[ij, kk, 2]
                v2_rr = u_prim_reshaped[ij, kk, 3]
                v3_rr = u_prim_reshaped[ij, kk, 4]
                p_rr = u_prim_reshaped[ij, kk, 5]

                # Compute required mean values
                rho_avg = 0.5 * (rho_ll + rho_rr)
                v1_avg = 0.5 * (v1_ll + v1_rr)
                v2_avg = 0.5 * (v2_ll + v2_rr)
                v3_avg = 0.5 * (v3_ll + v3_rr)
                p_avg = 0.5 * (p_ll + p_rr)
                kin_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)
                pv3_avg = 0.5 * (p_ll * v3_rr + p_rr * v3_ll)

                # Calculate fluxes depending on Cartesian orientation
                f1 = rho_avg * v3_avg
                f2 = f1 * v1_avg
                f3 = f1 * v2_avg
                f4 = f1 * v3_avg + p_avg
                f5 = p_avg * v3_avg * equations.inv_gamma_minus_one + f1 * kin_avg + pv3_avg

                # Add scaled fluxes to RHS
                factor_k = alpha * derivative_split[k, kk]
                du_reshaped[ij, k, 1] += factor_k * f1
                du_reshaped[ij, k, 2] += factor_k * f2
                du_reshaped[ij, k, 3] += factor_k * f3
                du_reshaped[ij, k, 4] += factor_k * f4
                du_reshaped[ij, k, 5] += factor_k * f5

                factor_kk = alpha * derivative_split[kk, k]
                du_reshaped[ij, kk, 1] += factor_kk * f1
                du_reshaped[ij, kk, 2] += factor_kk * f2
                du_reshaped[ij, kk, 3] += factor_kk * f3
                du_reshaped[ij, kk, 4] += factor_kk * f4
                du_reshaped[ij, kk, 5] += factor_kk * f5
            end
        end
    end # GC.@preserve u_prim

    # Finally, we add the temporary RHS computed here to the global RHS in the
    # given `element`.
    @turbo for v in eachvariable(equations),
               k in eachnode(dg),
               j in eachnode(dg),
               i in eachnode(dg)

        _du[v, i, j, k, element] += du[i, j, k, v]
    end
end

@inline function flux_differencing_kernel!(_du::PtrArray, u_cons::PtrArray,
                                           element, mesh::TreeMesh{3},
                                           nonconservative_terms::False,
                                           equations::CompressibleEulerEquations3D,
                                           volume_flux::typeof(flux_ranocha_turbo),
                                           dg::DGSEM, cache, alpha)
    @unpack derivative_split = dg.basis

    # Create a temporary array that will be used to store the RHS with permuted
    # indices `[i, j, k, v]` to allow using SIMD instructions.
    # `StrideArray`s with purely static dimensions do not allocate on the heap.
    du = StrideArray{eltype(u_cons)}(undef,
                                     (ntuple(_ -> StaticInt(nnodes(dg)), ndims(mesh))...,
                                      StaticInt(nvariables(equations))))

    # Convert conserved to primitive variables on the given `element`. In addition
    # to the usual primitive variables, we also compute logarithms of the density
    # and pressure to increase the performance of the required logarithmic mean
    # values.
    u_prim = StrideArray{eltype(u_cons)}(undef,
                                         (ntuple(_ -> StaticInt(nnodes(dg)),
                                                 ndims(mesh))...,
                                          StaticInt(nvariables(equations) + 2))) # We also compute "+ 2" logs

    @turbo for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        rho = u_cons[1, i, j, k, element]
        rho_v1 = u_cons[2, i, j, k, element]
        rho_v2 = u_cons[3, i, j, k, element]
        rho_v3 = u_cons[4, i, j, k, element]
        rho_e = u_cons[5, i, j, k, element]

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        p = (equations.gamma - 1) *
            (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3))

        u_prim[i, j, k, 1] = rho
        u_prim[i, j, k, 2] = v1
        u_prim[i, j, k, 3] = v2
        u_prim[i, j, k, 4] = v3
        u_prim[i, j, k, 5] = p
        u_prim[i, j, k, 6] = log(rho)
        u_prim[i, j, k, 7] = log(p)
    end

    # x direction
    # At first, we create new temporary arrays with permuted memory layout to
    # allow using SIMD instructions along the first dimension (which is contiguous
    # in memory).
    du_permuted = StrideArray{eltype(u_cons)}(undef,
                                              (StaticInt(nnodes(dg)^2),
                                               StaticInt(nnodes(dg)),
                                               StaticInt(nvariables(equations))))

    u_prim_permuted = StrideArray{eltype(u_cons)}(undef,
                                                  (StaticInt(nnodes(dg)^2),
                                                   StaticInt(nnodes(dg)),
                                                   StaticInt(nvariables(equations) + 2)))

    @turbo for v in indices(u_prim, 4), # v in eachvariable(equations) misses +2 logs
               k in eachnode(dg),
               j in eachnode(dg),
               i in eachnode(dg)

        jk = j + nnodes(dg) * (k - 1)
        u_prim_permuted[jk, i, v] = u_prim[i, j, k, v]
    end
    fill!(du_permuted, zero(eltype(du_permuted)))

    # Next, we basically inline the volume flux. To allow SIMD vectorization and
    # still use the symmetry of the volume flux and the derivative matrix, we
    # loop over the triangular part in an outer loop and use a plain inner loop.
    for i in eachnode(dg), ii in (i + 1):nnodes(dg)
        @turbo for jk in Base.OneTo(nnodes(dg)^2)
            rho_ll = u_prim_permuted[jk, i, 1]
            v1_ll = u_prim_permuted[jk, i, 2]
            v2_ll = u_prim_permuted[jk, i, 3]
            v3_ll = u_prim_permuted[jk, i, 4]
            p_ll = u_prim_permuted[jk, i, 5]
            log_rho_ll = u_prim_permuted[jk, i, 6]
            log_p_ll = u_prim_permuted[jk, i, 7]

            rho_rr = u_prim_permuted[jk, ii, 1]
            v1_rr = u_prim_permuted[jk, ii, 2]
            v2_rr = u_prim_permuted[jk, ii, 3]
            v3_rr = u_prim_permuted[jk, ii, 4]
            p_rr = u_prim_permuted[jk, ii, 5]
            log_rho_rr = u_prim_permuted[jk, ii, 6]
            log_p_rr = u_prim_permuted[jk, ii, 7]

            # Compute required mean values
            # We inline the logarithmic mean to allow LoopVectorization.jl to optimize
            # it efficiently. This is equivalent to
            #   rho_mean = ln_mean(rho_ll, rho_rr)
            x1 = rho_ll
            log_x1 = log_rho_ll
            y1 = rho_rr
            log_y1 = log_rho_rr
            x1_plus_y1 = x1 + y1
            y1_minus_x1 = y1 - x1
            z1 = y1_minus_x1^2 / x1_plus_y1^2
            special_path1 = x1_plus_y1 / (2 + z1 * (2 / 3 + z1 * (2 / 5 + 2 / 7 * z1)))
            regular_path1 = y1_minus_x1 / (log_y1 - log_x1)
            rho_mean = ifelse(z1 < 1.0e-4, special_path1, regular_path1)

            # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
            # in exact arithmetic since
            #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
            #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
            # inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
            x2 = rho_ll * p_rr
            log_x2 = log_rho_ll + log_p_rr
            y2 = rho_rr * p_ll
            log_y2 = log_rho_rr + log_p_ll
            x2_plus_y2 = x2 + y2
            y2_minus_x2 = y2 - x2
            z2 = y2_minus_x2^2 / x2_plus_y2^2
            special_path2 = (2 + z2 * (2 / 3 + z2 * (2 / 5 + 2 / 7 * z2))) / x2_plus_y2
            regular_path2 = (log_y2 - log_x2) / y2_minus_x2
            inv_rho_p_mean = p_ll * p_rr * ifelse(z2 < 1.0e-4, special_path2, regular_path2)

            v1_avg = 0.5 * (v1_ll + v1_rr)
            v2_avg = 0.5 * (v2_ll + v2_rr)
            v3_avg = 0.5 * (v3_ll + v3_rr)
            p_avg = 0.5 * (p_ll + p_rr)
            velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)

            # Calculate fluxes depending on Cartesian orientation
            f1 = rho_mean * v1_avg
            f2 = f1 * v1_avg + p_avg
            f3 = f1 * v2_avg
            f4 = f1 * v3_avg
            f5 = f1 *
                 (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one) +
                 0.5 * (p_ll * v1_rr + p_rr * v1_ll)

            # Add scaled fluxes to RHS
            factor_i = alpha * derivative_split[i, ii]
            du_permuted[jk, i, 1] += factor_i * f1
            du_permuted[jk, i, 2] += factor_i * f2
            du_permuted[jk, i, 3] += factor_i * f3
            du_permuted[jk, i, 4] += factor_i * f4
            du_permuted[jk, i, 5] += factor_i * f5

            factor_ii = alpha * derivative_split[ii, i]
            du_permuted[jk, ii, 1] += factor_ii * f1
            du_permuted[jk, ii, 2] += factor_ii * f2
            du_permuted[jk, ii, 3] += factor_ii * f3
            du_permuted[jk, ii, 4] += factor_ii * f4
            du_permuted[jk, ii, 5] += factor_ii * f5
        end
    end

    @turbo for v in eachvariable(equations),
               k in eachnode(dg),
               j in eachnode(dg),
               i in eachnode(dg)

        jk = j + nnodes(dg) * (k - 1)
        du[i, j, k, v] = du_permuted[jk, i, v]
    end

    # y direction
    # A possible permutation of array dimensions with improved opportunities for
    # SIMD vectorization appeared to be slower than the direct version used here
    # in preliminary numerical experiments on an AVX2 system.
    for j in eachnode(dg), jj in (j + 1):nnodes(dg)
        @turbo for k in eachnode(dg), i in eachnode(dg)
            rho_ll = u_prim[i, j, k, 1]
            v1_ll = u_prim[i, j, k, 2]
            v2_ll = u_prim[i, j, k, 3]
            v3_ll = u_prim[i, j, k, 4]
            p_ll = u_prim[i, j, k, 5]
            log_rho_ll = u_prim[i, j, k, 6]
            log_p_ll = u_prim[i, j, k, 7]

            rho_rr = u_prim[i, jj, k, 1]
            v1_rr = u_prim[i, jj, k, 2]
            v2_rr = u_prim[i, jj, k, 3]
            v3_rr = u_prim[i, jj, k, 4]
            p_rr = u_prim[i, jj, k, 5]
            log_rho_rr = u_prim[i, jj, k, 6]
            log_p_rr = u_prim[i, jj, k, 7]

            # Compute required mean values
            # We inline the logarithmic mean to allow LoopVectorization.jl to optimize
            # it efficiently. This is equivalent to
            #   rho_mean = ln_mean(rho_ll, rho_rr)
            x1 = rho_ll
            log_x1 = log_rho_ll
            y1 = rho_rr
            log_y1 = log_rho_rr
            x1_plus_y1 = x1 + y1
            y1_minus_x1 = y1 - x1
            z1 = y1_minus_x1^2 / x1_plus_y1^2
            special_path1 = x1_plus_y1 / (2 + z1 * (2 / 3 + z1 * (2 / 5 + 2 / 7 * z1)))
            regular_path1 = y1_minus_x1 / (log_y1 - log_x1)
            rho_mean = ifelse(z1 < 1.0e-4, special_path1, regular_path1)

            # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
            # in exact arithmetic since
            #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
            #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
            # inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
            x2 = rho_ll * p_rr
            log_x2 = log_rho_ll + log_p_rr
            y2 = rho_rr * p_ll
            log_y2 = log_rho_rr + log_p_ll
            x2_plus_y2 = x2 + y2
            y2_minus_x2 = y2 - x2
            z2 = y2_minus_x2^2 / x2_plus_y2^2
            special_path2 = (2 + z2 * (2 / 3 + z2 * (2 / 5 + 2 / 7 * z2))) / x2_plus_y2
            regular_path2 = (log_y2 - log_x2) / y2_minus_x2
            inv_rho_p_mean = p_ll * p_rr * ifelse(z2 < 1.0e-4, special_path2, regular_path2)

            v1_avg = 0.5 * (v1_ll + v1_rr)
            v2_avg = 0.5 * (v2_ll + v2_rr)
            v3_avg = 0.5 * (v3_ll + v3_rr)
            p_avg = 0.5 * (p_ll + p_rr)
            velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)

            # Calculate fluxes depending on Cartesian orientation
            f1 = rho_mean * v2_avg
            f2 = f1 * v1_avg
            f3 = f1 * v2_avg + p_avg
            f4 = f1 * v3_avg
            f5 = f1 *
                 (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one) +
                 0.5 * (p_ll * v2_rr + p_rr * v2_ll)

            # Add scaled fluxes to RHS
            factor_j = alpha * derivative_split[j, jj]
            du[i, j, k, 1] += factor_j * f1
            du[i, j, k, 2] += factor_j * f2
            du[i, j, k, 3] += factor_j * f3
            du[i, j, k, 4] += factor_j * f4
            du[i, j, k, 5] += factor_j * f5

            factor_jj = alpha * derivative_split[jj, j]
            du[i, jj, k, 1] += factor_jj * f1
            du[i, jj, k, 2] += factor_jj * f2
            du[i, jj, k, 3] += factor_jj * f3
            du[i, jj, k, 4] += factor_jj * f4
            du[i, jj, k, 5] += factor_jj * f5
        end
    end

    # z direction
    # The memory layout is already optimal for SIMD vectorization in this loop.
    # We just squeeze the first two dimensions to make the code slightly faster.
    GC.@preserve u_prim begin
        u_prim_reshaped = PtrArray(pointer(u_prim),
                                   (StaticInt(nnodes(dg)^2), StaticInt(nnodes(dg)),
                                    StaticInt(nvariables(equations) + 2)))

        du_reshaped = PtrArray(pointer(du),
                               (StaticInt(nnodes(dg)^2), StaticInt(nnodes(dg)),
                                StaticInt(nvariables(equations))))

        for k in eachnode(dg), kk in (k + 1):nnodes(dg)
            @turbo for ij in Base.OneTo(nnodes(dg)^2)
                rho_ll = u_prim_reshaped[ij, k, 1]
                v1_ll = u_prim_reshaped[ij, k, 2]
                v2_ll = u_prim_reshaped[ij, k, 3]
                v3_ll = u_prim_reshaped[ij, k, 4]
                p_ll = u_prim_reshaped[ij, k, 5]
                log_rho_ll = u_prim_reshaped[ij, k, 6]
                log_p_ll = u_prim_reshaped[ij, k, 7]

                rho_rr = u_prim_reshaped[ij, kk, 1]
                v1_rr = u_prim_reshaped[ij, kk, 2]
                v2_rr = u_prim_reshaped[ij, kk, 3]
                v3_rr = u_prim_reshaped[ij, kk, 4]
                p_rr = u_prim_reshaped[ij, kk, 5]
                log_rho_rr = u_prim_reshaped[ij, kk, 6]
                log_p_rr = u_prim_reshaped[ij, kk, 7]

                # Compute required mean values
                # We inline the logarithmic mean to allow LoopVectorization.jl to optimize
                # it efficiently. This is equivalent to
                #   rho_mean = ln_mean(rho_ll, rho_rr)
                x1 = rho_ll
                log_x1 = log_rho_ll
                y1 = rho_rr
                log_y1 = log_rho_rr
                x1_plus_y1 = x1 + y1
                y1_minus_x1 = y1 - x1
                z1 = y1_minus_x1^2 / x1_plus_y1^2
                special_path1 = x1_plus_y1 / (2 + z1 * (2 / 3 + z1 * (2 / 5 + 2 / 7 * z1)))
                regular_path1 = y1_minus_x1 / (log_y1 - log_x1)
                rho_mean = ifelse(z1 < 1.0e-4, special_path1, regular_path1)

                # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
                # in exact arithmetic since
                #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
                #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
                # inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
                x2 = rho_ll * p_rr
                log_x2 = log_rho_ll + log_p_rr
                y2 = rho_rr * p_ll
                log_y2 = log_rho_rr + log_p_ll
                x2_plus_y2 = x2 + y2
                y2_minus_x2 = y2 - x2
                z2 = y2_minus_x2^2 / x2_plus_y2^2
                special_path2 = (2 + z2 * (2 / 3 + z2 * (2 / 5 + 2 / 7 * z2))) / x2_plus_y2
                regular_path2 = (log_y2 - log_x2) / y2_minus_x2
                inv_rho_p_mean = p_ll * p_rr *
                                 ifelse(z2 < 1.0e-4, special_path2, regular_path2)

                v1_avg = 0.5 * (v1_ll + v1_rr)
                v2_avg = 0.5 * (v2_ll + v2_rr)
                v3_avg = 0.5 * (v3_ll + v3_rr)
                p_avg = 0.5 * (p_ll + p_rr)
                velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)

                # Calculate fluxes depending on Cartesian orientation
                f1 = rho_mean * v3_avg
                f2 = f1 * v1_avg
                f3 = f1 * v2_avg
                f4 = f1 * v3_avg + p_avg
                f5 = f1 *
                     (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one) +
                     0.5 * (p_ll * v3_rr + p_rr * v3_ll)

                # Add scaled fluxes to RHS
                factor_k = alpha * derivative_split[k, kk]
                du_reshaped[ij, k, 1] += factor_k * f1
                du_reshaped[ij, k, 2] += factor_k * f2
                du_reshaped[ij, k, 3] += factor_k * f3
                du_reshaped[ij, k, 4] += factor_k * f4
                du_reshaped[ij, k, 5] += factor_k * f5

                factor_kk = alpha * derivative_split[kk, k]
                du_reshaped[ij, kk, 1] += factor_kk * f1
                du_reshaped[ij, kk, 2] += factor_kk * f2
                du_reshaped[ij, kk, 3] += factor_kk * f3
                du_reshaped[ij, kk, 4] += factor_kk * f4
                du_reshaped[ij, kk, 5] += factor_kk * f5
            end
        end
    end # GC.@preserve u_prim

    # Finally, we add the temporary RHS computed here to the global RHS in the
    # given `element`.
    @turbo for v in eachvariable(equations),
               k in eachnode(dg),
               j in eachnode(dg),
               i in eachnode(dg)

        _du[v, i, j, k, element] += du[i, j, k, v]
    end
end
