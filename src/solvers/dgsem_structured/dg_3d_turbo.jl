@inline function flux_differencing_kernel!(_du::PtrArray, u_cons::PtrArray, element,
                                           MeshT::Type{<:Union{StructuredMesh{3},
                                                               P4estMesh{3}}},
                                           have_nonconservative_terms::False,
                                           equations,
                                           volume_flux::FluxTurbo, dg::DGSEM,
                                           cache, alpha)
    @unpack numerical_flux = volume_flux
    flux_differencing_kernel_turbo!(_du, u_cons, element, MeshT, have_nonconservative_terms,
                                    equations,
                                    numerical_flux, dg, cache, alpha,
                                    nturbovars(numerical_flux, equations),
                                    Val(nvariables(equations)))
end

# Generated function that writes the equivalent hand-written code as in dg_compressible_euler_3d.jl,
# but generalizes for each number of variables and precomputed variables.
@generated function flux_differencing_kernel_turbo!(_du::PtrArray, u_cons::PtrArray,
                                                    element,
                                                    MeshT::Type{<:Union{StructuredMesh{3},
                                                                        P4estMesh{3}}},
                                                    have_nonconservative_terms::False,
                                                    equations,
                                                    volume_flux,
                                                    dg, cache, alpha, ::Val{NAUX},
                                                    ::Val{NVARS}) where {NAUX, NVARS}
    # Per-node scalars used inside the `@turbo` loops, e.g. `u_prim_ll_1 = rho_ll`,
    # `flux_1 = f1`, etc.  
    u_prim_ll = [Symbol(:u_prim_ll_, v) for v in 1:NAUX]
    u_prim_rr = [Symbol(:u_prim_rr_, v) for v in 1:NAUX]
    flux = [Symbol(:flux_, v) for v in 1:NVARS]

    # Convert conserved to auxiliary variables node-wise, e.g. 
    #   rho, v1, v2, v3, p = cons2prim(u_cons[:, i, j, k, element], equations)
    #   u_prim[i, j, k, 1] = rho   # and so on for v2, v3, p, ...
    cons_reads = [:(u_cons[$v, i, j, k, element]) for v in 1:NVARS]
    cons2turbo_ = Expr(:(=), Expr(:tuple, [Symbol(:u_prim_, v) for v in 1:NAUX]...),
                       :(cons2turbo(volume_flux, $(cons_reads...),
                                    equations)))
    cons2turbo_writes = [:(u_prim[i, j, k, $v] = $(Symbol(:u_prim_, v))) for v in 1:NAUX]

    # Evaluate the two-point volume flux
    flux_call = Expr(:(=), Expr(:tuple, flux...),
                     :(flux_turbo(volume_flux,
                                  $(u_prim_ll...), $(u_prim_rr...),
                                  normal_direction_1, normal_direction_2,
                                  normal_direction_3,
                                  equations)))

    # Build the inner `@turbo` loop body. For the x direction `loads` gives
    #   rho_ll = u_prim_permuted[jk, i, 1]   # left node, and so on
    #   rho_rr = u_prim_permuted[jk, ii, 1]  # right node
    loads(arr, idx_ll, idx_rr) = vcat([:($(u_prim_ll[v]) = $arr[$(idx_ll...), $v])
                                       for v in 1:NAUX],
                                      [:($(u_prim_rr[v]) = $arr[$(idx_rr...), $v])
                                       for v in 1:NAUX])

    # Compute the average of the normal_direction
    # e.g. the x direction `normals` gives
    #   normal_direction_1 = 0.5f0 * (contravariant_vectors_x[jk, i, 1] +
    #                                 contravariant_vectors_x[jk, ii, 1])   # and 2, 3
    normals(controvariant_vector, idx_ll, idx_rr) = [:($(Symbol(:normal_direction_, m)) = 0.5f0 *
                                                                                          ($controvariant_vector[$(idx_ll...),
                                                                                                                 $m] +
                                                                                           $controvariant_vector[$(idx_rr...),
                                                                                                                 $m]))
                                                     for m in 1:3]

    # Store the fluxes in the permuted du array
    # e.g. the x direction `stores` gives
    #   du_permuted[jk, i, 1]  += factor_l * f1   # left node, and so on
    #   du_permuted[jk, ii, 1] += factor_r * f1   # right node
    stores(du_arr, idx_ll, idx_rr) = vcat([:($du_arr[$(idx_ll...), $v] += factor_l *
                                                                          $(flux[v]))
                                           for v in 1:NVARS],
                                          [:($du_arr[$(idx_rr...), $v] += factor_r *
                                                                          $(flux[v]))
                                           for v in 1:NVARS])

    loads_x = loads(:u_prim_permuted, (:jk, :i), (:jk, :ii))
    normals_x = normals(:contravariant_vectors_x, (:jk, :i), (:jk, :ii))
    stores_x = stores(:du_permuted, (:jk, :i), (:jk, :ii))

    loads_y = loads(:u_prim, (:i, :j, :k), (:i, :jj, :k))
    normals_y = normals(:contravariant_vectors_y, (:i, :j, :k), (:i, :jj, :k))
    stores_y = stores(:du, (:i, :j, :k), (:i, :jj, :k))

    loads_z = loads(:u_prim_reshaped, (:ij, :k), (:ij, :kk))
    normals_z = normals(:contravariant_vectors_z, (:ij, :k), (:ij, :kk))
    stores_z = stores(:du_reshaped, (:ij, :k), (:ij, :kk))

    quote
        @unpack derivative_split = dg.basis
        @unpack contravariant_vectors = cache.elements

        # Create a temporary array that will be used to store the RHS with permuted
        # indices `[i, j, k, v]` to allow using SIMD instructions.
        # `StrideArray`s with purely static dimensions do not allocate on the heap.
        du = StrideArray{eltype(u_cons)}(undef,
                                         (StaticInt(nnodes(dg)), StaticInt(nnodes(dg)),
                                          StaticInt(nnodes(dg)), StaticInt(NVARS)))

        # Convert conserved to auxiliary variables on the given `element`.
        u_prim = StrideArray{eltype(u_cons)}(undef,
                                             (StaticInt(nnodes(dg)), StaticInt(nnodes(dg)),
                                              StaticInt(nnodes(dg)), StaticInt(NAUX)))

        # Convert conserved to auxiliary variables and store them in `u_prim`.
        @turbo for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            $cons2turbo_
            $(cons2turbo_writes...)
        end

        # x direction
        # At first, we create new temporary arrays with permuted memory layout to
        # allow using SIMD instructions along the first dimension (which is contiguous
        # in memory).
        du_permuted = StrideArray{eltype(u_cons)}(undef,
                                                  (StaticInt(nnodes(dg)^2),
                                                   StaticInt(nnodes(dg)),
                                                   StaticInt(NVARS)))

        u_prim_permuted = StrideArray{eltype(u_cons)}(undef,
                                                      (StaticInt(nnodes(dg)^2),
                                                       StaticInt(nnodes(dg)),
                                                       StaticInt(NAUX)))

        @turbo for v in indices(u_prim, 4),
                   k in eachnode(dg),
                   j in eachnode(dg),
                   i in eachnode(dg)

            jk = j + nnodes(dg) * (k - 1)
            u_prim_permuted[jk, i, v] = u_prim[i, j, k, v]
        end
        fill!(du_permuted, zero(eltype(du_permuted)))

        # We must also permute the contravariant vectors.
        contravariant_vectors_x = StrideArray{eltype(contravariant_vectors)}(undef,
                                                                             (StaticInt(nnodes(dg)^2),
                                                                              StaticInt(nnodes(dg)),
                                                                              StaticInt(3)))

        @turbo for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            jk = j + nnodes(dg) * (k - 1)
            contravariant_vectors_x[jk, i, 1] = contravariant_vectors[1, 1, i, j, k,
                                                                      element]
            contravariant_vectors_x[jk, i, 2] = contravariant_vectors[2, 1, i, j, k,
                                                                      element]
            contravariant_vectors_x[jk, i, 3] = contravariant_vectors[3, 1, i, j, k,
                                                                      element]
        end

        # Next, we basically inline the volume flux. To allow SIMD vectorization and
        # still use the symmetry of the volume flux and the derivative matrix, we
        # loop over the triangular part in an outer loop and use a plain inner loop.
        for i in eachnode(dg), ii in (i + 1):nnodes(dg)
            @turbo for jk in Base.OneTo(nnodes(dg)^2)
                $(loads_x...)
                $(normals_x...)
                $flux_call
                factor_l = alpha * derivative_split[i, ii]
                factor_r = alpha * derivative_split[ii, i]
                $(stores_x...)
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
        # We must also permute the contravariant vectors.
        contravariant_vectors_y = StrideArray{eltype(contravariant_vectors)}(undef,
                                                                             (StaticInt(nnodes(dg)),
                                                                              StaticInt(nnodes(dg)),
                                                                              StaticInt(nnodes(dg)),
                                                                              StaticInt(3)))

        @turbo for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            contravariant_vectors_y[i, j, k, 1] = contravariant_vectors[1, 2, i, j, k,
                                                                        element]
            contravariant_vectors_y[i, j, k, 2] = contravariant_vectors[2, 2, i, j, k,
                                                                        element]
            contravariant_vectors_y[i, j, k, 3] = contravariant_vectors[3, 2, i, j, k,
                                                                        element]
        end

        # A possible permutation of array dimensions with improved opportunities for
        # SIMD vectorization appeared to be slower than the direct version used here
        # in preliminary numerical experiments on an AVX2 system.
        for j in eachnode(dg), jj in (j + 1):nnodes(dg)
            @turbo for k in eachnode(dg), i in eachnode(dg)
                $(loads_y...)
                $(normals_y...)
                $flux_call
                factor_l = alpha * derivative_split[j, jj]
                factor_r = alpha * derivative_split[jj, j]
                $(stores_y...)
            end
        end

        # z direction
        # The memory layout is already optimal for SIMD vectorization in this loop.
        # We just squeeze the first two dimensions to make the code slightly faster.
        GC.@preserve u_prim begin
            u_prim_reshaped = PtrArray(pointer(u_prim),
                                       (StaticInt(nnodes(dg)^2), StaticInt(nnodes(dg)),
                                        StaticInt(NAUX)))

            du_reshaped = PtrArray(pointer(du),
                                   (StaticInt(nnodes(dg)^2), StaticInt(nnodes(dg)),
                                    StaticInt(NVARS)))

            # We must also permute the contravariant vectors.
            contravariant_vectors_z = StrideArray{eltype(contravariant_vectors)}(undef,
                                                                                 (StaticInt(nnodes(dg)^2),
                                                                                  StaticInt(nnodes(dg)),
                                                                                  StaticInt(3)))

            @turbo for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                ij = i + nnodes(dg) * (j - 1)
                contravariant_vectors_z[ij, k, 1] = contravariant_vectors[1, 3, i, j, k,
                                                                          element]
                contravariant_vectors_z[ij, k, 2] = contravariant_vectors[2, 3, i, j, k,
                                                                          element]
                contravariant_vectors_z[ij, k, 3] = contravariant_vectors[3, 3, i, j, k,
                                                                          element]
            end

            for k in eachnode(dg), kk in (k + 1):nnodes(dg)
                @turbo for ij in Base.OneTo(nnodes(dg)^2)
                    $(loads_z...)
                    $(normals_z...)
                    $flux_call
                    factor_l = alpha * derivative_split[k, kk]
                    factor_r = alpha * derivative_split[kk, k]
                    $(stores_z...)
                end
            end
        end # GC.@preserve u_prim begin

        # Finally, we add the temporary RHS computed here to the global RHS in the
        # given `element`.
        @turbo for v in eachvariable(equations),
                   k in eachnode(dg),
                   j in eachnode(dg),
                   i in eachnode(dg)

            _du[v, i, j, k, element] += du[i, j, k, v]
        end

        return nothing
    end
end

@inline function flux_differencing_kernel!(_du::PtrArray, u_cons::PtrArray, element,
                                           MeshT::Type{<:Union{StructuredMesh{3},
                                                               P4estMesh{3}}},
                                           have_nonconservative_terms::True,
                                           equations,
                                           volume_flux::FluxTurbo, dg::DGSEM,
                                           cache, alpha)
    @unpack numerical_flux = volume_flux
    flux_differencing_kernel_turbo!(_du, u_cons, element, MeshT, have_nonconservative_terms,
                                    equations,
                                    volume_flux, dg, cache, alpha,
                                    nturbovars(numerical_flux..., equations),
                                    Val(nvariables(equations)))
end

@generated function flux_differencing_kernel_turbo!(_du::PtrArray, u_cons::PtrArray,
                                                    element,
                                                    MeshT::Type{<:Union{StructuredMesh{3},
                                                                        P4estMesh{3}}},
                                                    have_nonconservative_terms::True,
                                                    equations,
                                                    volume_flux,
                                                    dg, cache, alpha, ::Val{NAUX},
                                                    ::Val{NVARS}) where {NAUX, NVARS}
    # Per-node scalars used inside the `@turbo` loops, e.g. `u_prim_ll_1 = rho_ll`,
    # `flux_1 = f1, etc.  
    u_prim_ll = [Symbol(:u_prim_ll_, v) for v in 1:NAUX]
    u_prim_rr = [Symbol(:u_prim_rr_, v) for v in 1:NAUX]
    flux_left = [Symbol(:flux_left_, v) for v in 1:NVARS]
    flux_right = [Symbol(:flux_right_, v) for v in 1:NVARS]

    # Convert conserved to auxiliary variables node-wise, e.g. 
    #   rho, v1, v2, v3, p = cons2prim(u_cons[:, i, j, k, element], equations)
    #   u_prim[i, j, k, 1] = rho   # and so on for v2, v3, p, ...
    cons_reads = [:(u_cons[$v, i, j, k, element]) for v in 1:NVARS]
    cons2turbo_ = Expr(:(=), Expr(:tuple, [Symbol(:u_prim_, v) for v in 1:NAUX]...),
                       :(cons2turbo(volume_flux, $(cons_reads...),
                                    equations)))
    cons2turbo_writes = [:(u_prim[i, j, k, $v] = $(Symbol(:u_prim_, v))) for v in 1:NAUX]

    # Evaluate the two-point volume flux
    flux_call = Expr(:(=),
                     Expr(:tuple, Expr(:tuple, flux_left...), Expr(:tuple, flux_right...)),
                     :(flux_turbo(volume_flux,
                                  $(u_prim_ll...),
                                  $(u_prim_rr...),
                                  normal_direction_1, normal_direction_2,
                                  normal_direction_3, equations)))

    # Build the inner `@turbo` loop body. For the x direction `loads` gives
    #   rho_ll = u_prim_permuted[jk, i, 1]   # left node, and so on
    #   rho_rr = u_prim_permuted[jk, ii, 1]  # right node
    loads(arr, idx_ll, idx_rr) = vcat([:($(u_prim_ll[v]) = $arr[$(idx_ll...), $v])
                                       for v in 1:NAUX],
                                      [:($(u_prim_rr[v]) = $arr[$(idx_rr...), $v])
                                       for v in 1:NAUX])

    # Compute the average of the normal_direction
    # e.g. the x direction `normals` gives
    #   normal_direction_1 = 0.5f0 * (contravariant_vectors_x[jk, i, 1] +
    #                                 contravariant_vectors_x[jk, ii, 1])   # and 2, 3
    normals(controvariant_vector, idx_ll, idx_rr) = [:($(Symbol(:normal_direction_, m)) = 0.5f0 *
                                                                                          ($controvariant_vector[$(idx_ll...),
                                                                                                                 $m] +
                                                                                           $controvariant_vector[$(idx_rr...),
                                                                                                                 $m]))
                                                     for m in 1:3]

    # Store the fluxes in the permuted du array
    # e.g. the x direction `stores` gives
    #   du_permuted[jk, i, 1]  += factor_l * f1   # left node, and so on
    #   du_permuted[jk, ii, 1] += factor_r * f1   # right node
    stores(du_arr, idx_ll, idx_rr) = vcat([:($du_arr[$(idx_ll...), $v] += factor_l *
                                                                          $(flux_left[v]))
                                           for v in 1:NVARS],
                                          [:($du_arr[$(idx_rr...), $v] += factor_r *
                                                                          $(flux_right[v]))
                                           for v in 1:NVARS])

    loads_x = loads(:u_prim_permuted, (:jk, :i), (:jk, :ii))
    normals_x = normals(:contravariant_vectors_x, (:jk, :i), (:jk, :ii))
    stores_x = stores(:du_permuted, (:jk, :i), (:jk, :ii))

    loads_y = loads(:u_prim, (:i, :j, :k), (:i, :jj, :k))
    normals_y = normals(:contravariant_vectors_y, (:i, :j, :k), (:i, :jj, :k))
    stores_y = stores(:du, (:i, :j, :k), (:i, :jj, :k))

    loads_z = loads(:u_prim_reshaped, (:ij, :k), (:ij, :kk))
    normals_z = normals(:contravariant_vectors_z, (:ij, :k), (:ij, :kk))
    stores_z = stores(:du_reshaped, (:ij, :k), (:ij, :kk))

    quote
        @unpack derivative_split = dg.basis
        @unpack contravariant_vectors = cache.elements

        # Create a temporary array that will be used to store the RHS with permuted
        # indices `[i, j, k, v]` to allow using SIMD instructions.
        # `StrideArray`s with purely static dimensions do not allocate on the heap.
        du = StrideArray{eltype(u_cons)}(undef,
                                         (StaticInt(nnodes(dg)), StaticInt(nnodes(dg)),
                                          StaticInt(nnodes(dg)), StaticInt(NVARS)))

        # Convert conserved to auxiliary variables on the given `element`.
        u_prim = StrideArray{eltype(u_cons)}(undef,
                                             (StaticInt(nnodes(dg)), StaticInt(nnodes(dg)),
                                              StaticInt(nnodes(dg)), StaticInt(NAUX)))

        # Convert conserved to auxiliary variables and store them in `u_prim`.
        @turbo for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            $cons2turbo_
            $(cons2turbo_writes...)
        end

        # x direction
        # At first, we create new temporary arrays with permuted memory layout to
        # allow using SIMD instructions along the first dimension (which is contiguous
        # in memory).
        du_permuted = StrideArray{eltype(u_cons)}(undef,
                                                  (StaticInt(nnodes(dg)^2),
                                                   StaticInt(nnodes(dg)),
                                                   StaticInt(NVARS)))

        u_prim_permuted = StrideArray{eltype(u_cons)}(undef,
                                                      (StaticInt(nnodes(dg)^2),
                                                       StaticInt(nnodes(dg)),
                                                       StaticInt(NAUX)))

        @turbo for v in indices(u_prim, 4),
                   k in eachnode(dg),
                   j in eachnode(dg),
                   i in eachnode(dg)

            jk = j + nnodes(dg) * (k - 1)
            u_prim_permuted[jk, i, v] = u_prim[i, j, k, v]
        end
        fill!(du_permuted, zero(eltype(du_permuted)))

        # We must also permute the contravariant vectors.
        contravariant_vectors_x = StrideArray{eltype(contravariant_vectors)}(undef,
                                                                             (StaticInt(nnodes(dg)^2),
                                                                              StaticInt(nnodes(dg)),
                                                                              StaticInt(3)))

        @turbo for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            jk = j + nnodes(dg) * (k - 1)
            contravariant_vectors_x[jk, i, 1] = contravariant_vectors[1, 1, i, j, k,
                                                                      element]
            contravariant_vectors_x[jk, i, 2] = contravariant_vectors[2, 1, i, j, k,
                                                                      element]
            contravariant_vectors_x[jk, i, 3] = contravariant_vectors[3, 1, i, j, k,
                                                                      element]
        end

        # Next, we basically inline the volume flux. To allow SIMD vectorization and
        # still use the symmetry of the volume flux and the derivative matrix, we
        # loop over the triangular part in an outer loop and use a plain inner loop.
        for i in eachnode(dg), ii in (i + 1):nnodes(dg)
            @turbo for jk in Base.OneTo(nnodes(dg)^2)
                $(loads_x...)
                $(normals_x...)
                $flux_call
                factor_l = alpha * derivative_split[i, ii]
                factor_r = alpha * derivative_split[ii, i]
                $(stores_x...)
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
        # We must also permute the contravariant vectors.
        contravariant_vectors_y = StrideArray{eltype(contravariant_vectors)}(undef,
                                                                             (StaticInt(nnodes(dg)),
                                                                              StaticInt(nnodes(dg)),
                                                                              StaticInt(nnodes(dg)),
                                                                              StaticInt(3)))

        @turbo for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            contravariant_vectors_y[i, j, k, 1] = contravariant_vectors[1, 2, i, j, k,
                                                                        element]
            contravariant_vectors_y[i, j, k, 2] = contravariant_vectors[2, 2, i, j, k,
                                                                        element]
            contravariant_vectors_y[i, j, k, 3] = contravariant_vectors[3, 2, i, j, k,
                                                                        element]
        end

        # A possible permutation of array dimensions with improved opportunities for
        # SIMD vectorization appeared to be slower than the direct version used here
        # in preliminary numerical experiments on an AVX2 system.
        for j in eachnode(dg), jj in (j + 1):nnodes(dg)
            @turbo for k in eachnode(dg), i in eachnode(dg)
                $(loads_y...)
                $(normals_y...)
                $flux_call
                factor_l = alpha * derivative_split[j, jj]
                factor_r = alpha * derivative_split[jj, j]
                $(stores_y...)
            end
        end

        # z direction
        # The memory layout is already optimal for SIMD vectorization in this loop.
        # We just squeeze the first two dimensions to make the code slightly faster.
        GC.@preserve u_prim begin
            u_prim_reshaped = PtrArray(pointer(u_prim),
                                       (StaticInt(nnodes(dg)^2), StaticInt(nnodes(dg)),
                                        StaticInt(NAUX)))

            du_reshaped = PtrArray(pointer(du),
                                   (StaticInt(nnodes(dg)^2), StaticInt(nnodes(dg)),
                                    StaticInt(NVARS)))

            # We must also permute the contravariant vectors.
            contravariant_vectors_z = StrideArray{eltype(contravariant_vectors)}(undef,
                                                                                 (StaticInt(nnodes(dg)^2),
                                                                                  StaticInt(nnodes(dg)),
                                                                                  StaticInt(3)))

            @turbo for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                ij = i + nnodes(dg) * (j - 1)
                contravariant_vectors_z[ij, k, 1] = contravariant_vectors[1, 3, i, j, k,
                                                                          element]
                contravariant_vectors_z[ij, k, 2] = contravariant_vectors[2, 3, i, j, k,
                                                                          element]
                contravariant_vectors_z[ij, k, 3] = contravariant_vectors[3, 3, i, j, k,
                                                                          element]
            end

            for k in eachnode(dg), kk in (k + 1):nnodes(dg)
                @turbo for ij in Base.OneTo(nnodes(dg)^2)
                    $(loads_z...)
                    $(normals_z...)
                    $flux_call
                    factor_l = alpha * derivative_split[k, kk]
                    factor_r = alpha * derivative_split[kk, k]
                    $(stores_z...)
                end
            end
        end # GC.@preserve u_prim begin

        # Finally, we add the temporary RHS computed here to the global RHS in the
        # given `element`.
        @turbo for v in eachvariable(equations),
                   k in eachnode(dg),
                   j in eachnode(dg),
                   i in eachnode(dg)

            _du[v, i, j, k, element] += du[i, j, k, v]
        end

        return nothing
    end
end

# Number of precomputed variables for the specialization flux_ranocha
@inline nturbovars(flux_turbo::typeof(flux_ranocha), equations::CompressibleEulerEquations3D) = Val(7)

# Transformation from conserved to precomputed variables for flux_ranocha
@inline function cons2turbo(flux_turbo::typeof(flux_ranocha),
                            rho, rho_v1, rho_v2, rho_v3, rho_e,
                            equations::CompressibleEulerEquations3D)
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    p = (equations.gamma - 1) *
        (rho_e - 0.5f0 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3))
    return rho, v1, v2, v3, p, log(rho), log(p)
end

# Computation of the numerical flux_ranocha with respect to precomputed variables
@inline function flux_turbo(flux_turbo::typeof(flux_ranocha),
                            rho_ll, v1_ll, v2_ll, v3_ll,
                            p_ll, log_rho_ll, log_p_ll,
                            rho_rr, v1_rr, v2_rr, v3_rr,
                            p_rr, log_rho_rr, log_p_rr,
                            normal_direction_1,
                            normal_direction_2,
                            normal_direction_3,
                            equations::CompressibleEulerEquations3D)
    v_dot_n_ll = v1_ll * normal_direction_1 + v2_ll * normal_direction_2 +
                 v3_ll * normal_direction_3
    v_dot_n_rr = v1_rr * normal_direction_1 + v2_rr * normal_direction_2 +
                 v3_rr * normal_direction_3

    # Compute required mean values
    # We inline the logarithmic mean to allow LoopVectorization.jl to optimize
    # it efficiently. This is equivalent to
    # rho_mean = ln_mean(rho_ll, rho_rr)
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

    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v3_avg = 0.5f0 * (v3_ll + v3_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    velocity_square_avg = 0.5f0 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    f2 = f1 * v1_avg + p_avg * normal_direction_1
    f3 = f1 * v2_avg + p_avg * normal_direction_2
    f4 = f1 * v3_avg + p_avg * normal_direction_3
    f5 = (f1 * (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one)
          +
          0.5f0 * (p_ll * v_dot_n_rr + p_rr * v_dot_n_ll))

    return (f1, f2, f3, f4, f5)
end
