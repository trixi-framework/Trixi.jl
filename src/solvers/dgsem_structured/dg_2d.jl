# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function create_cache(mesh::Union{StructuredMesh{2}, UnstructuredMesh2D,
                                  P4estMesh{2}, T8codeMesh{2}}, equations,
                      volume_integral::Union{AbstractVolumeIntegralPureLGLFiniteVolume,
                                             VolumeIntegralShockCapturingHG},
                      dg::DG, cache_containers, uEltype)
    fstar1_L_threaded, fstar1_R_threaded,
    fstar2_L_threaded, fstar2_R_threaded = create_f_threaded(mesh, equations, dg,
                                                             uEltype)

    normal_vectors = NormalVectorContainer2D(mesh, dg, cache_containers)

    return (; fstar1_L_threaded, fstar1_R_threaded,
            fstar2_L_threaded, fstar2_R_threaded,
            normal_vectors)
end

#=
`weak_form_kernel!` is only implemented for conserved terms as
non-conservative terms should always be discretized in conjunction with a flux-splitting scheme,
see `flux_differencing_kernel!`.
This treatment is required to achieve, e.g., entropy-stability or well-balancedness.
See also https://github.com/trixi-framework/Trixi.jl/issues/1671#issuecomment-1765644064
=#
@inline function weak_form_kernel!(du, u,
                                   element,
                                   mesh::Union{StructuredMesh{2}, StructuredMeshView{2},
                                               UnstructuredMesh2D, P4estMesh{2},
                                               P4estMeshView{2}, T8codeMesh{2}},
                                   have_nonconservative_terms::False, equations,
                                   dg::DGSEM, cache, alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    @unpack derivative_dhat = dg.basis
    @unpack contravariant_vectors = cache.elements

    for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)

        flux1 = flux(u_node, 1, equations)
        flux2 = flux(u_node, 2, equations)

        # Compute the contravariant flux by taking the scalar product of the
        # first contravariant vector Ja^1 and the flux vector
        Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
        contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2
        for ii in eachnode(dg)
            multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i],
                                       contravariant_flux1, equations, dg,
                                       ii, j, element)
        end

        # Compute the contravariant flux by taking the scalar product of the
        # second contravariant vector Ja^2 and the flux vector
        Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
        contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2
        for jj in eachnode(dg)
            multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j],
                                       contravariant_flux2, equations, dg,
                                       i, jj, element)
        end
    end

    return nothing
end

@inline function flux_differencing_kernel!(du, u,
                                           element,
                                           mesh::Union{StructuredMesh{2},
                                                       StructuredMeshView{2},
                                                       UnstructuredMesh2D, P4estMesh{2},
                                                       T8codeMesh{2}},
                                           have_nonconservative_terms::False, equations,
                                           volume_flux, dg::DGSEM, cache, alpha = true)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements

    # Calculate volume integral in one element
    for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)

        # pull the contravariant vectors in each coordinate direction
        Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, element)
        Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.

        # x direction
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
            # pull the contravariant vectors and compute the average
            Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                                   ii, j, element)
            # average mapping terms in first coordinate direction,
            # used as normal vector in the flux computation
            Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
            # compute the contravariant sharp flux in the direction of the
            # averaged contravariant vector
            fluxtilde1 = volume_flux(u_node, u_node_ii, Ja1_avg, equations)
            multiply_add_to_node_vars!(du, alpha * derivative_split[i, ii], fluxtilde1,
                                       equations, dg, i, j, element)
            multiply_add_to_node_vars!(du, alpha * derivative_split[ii, i], fluxtilde1,
                                       equations, dg, ii, j, element)
        end

        # y direction
        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
            # pull the contravariant vectors and compute the average
            Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                                   i, jj, element)
            # average mapping terms in second coordinate direction,
            # used as normal vector in the flux computation
            Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
            # compute the contravariant sharp flux in the direction of the
            # averaged contravariant vector
            fluxtilde2 = volume_flux(u_node, u_node_jj, Ja2_avg, equations)
            multiply_add_to_node_vars!(du, alpha * derivative_split[j, jj], fluxtilde2,
                                       equations, dg, i, j, element)
            multiply_add_to_node_vars!(du, alpha * derivative_split[jj, j], fluxtilde2,
                                       equations, dg, i, jj, element)
        end
    end

    return nothing
end

@inline function flux_differencing_kernel!(du, u,
                                           element,
                                           mesh::Union{StructuredMesh{2},
                                                       StructuredMeshView{2},
                                                       UnstructuredMesh2D, P4estMesh{2},
                                                       T8codeMesh{2}},
                                           have_nonconservative_terms::True, equations,
                                           volume_flux, dg::DGSEM, cache, alpha = true)
    flux_differencing_kernel!(du, u, element, mesh, have_nonconservative_terms,
                              combine_conservative_and_nonconservative_fluxes(volume_flux,
                                                                              equations),
                              equations,
                              volume_flux,
                              dg, cache, alpha)
    return nothing
end

@inline function flux_differencing_kernel!(du, u,
                                           element,
                                           mesh::Union{StructuredMesh{2},
                                                       StructuredMeshView{2},
                                                       UnstructuredMesh2D, P4estMesh{2},
                                                       T8codeMesh{2}},
                                           have_nonconservative_terms::True,
                                           combine_conservative_and_nonconservative_fluxes::False,
                                           equations,
                                           volume_flux, dg::DGSEM, cache, alpha = true)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    symmetric_flux, nonconservative_flux = volume_flux

    # Apply the symmetric flux as usual
    flux_differencing_kernel!(du, u, element, mesh, False(), equations, symmetric_flux,
                              dg, cache, alpha)

    # Calculate the remaining volume terms using the nonsymmetric generalized flux
    for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)

        # pull the contravariant vectors in each coordinate direction
        Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, element)
        Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, element)

        # The diagonal terms are zero since the diagonal of `derivative_split`
        # is zero. We ignore this for now.
        # In general, nonconservative fluxes can depend on both the contravariant
        # vectors (normal direction) at the current node and the averaged ones.
        # Thus, we need to pass both to the nonconservative flux.

        # x direction
        integral_contribution = zero(u_node)
        for ii in eachnode(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
            # pull the contravariant vectors and compute the average
            Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                                   ii, j, element)
            # average mapping terms in first coordinate direction,
            # used as normal vector in the flux computation
            Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
            # Compute the contravariant nonconservative flux.
            fluxtilde1 = nonconservative_flux(u_node, u_node_ii, Ja1_avg,
                                              equations)
            integral_contribution = integral_contribution +
                                    derivative_split[i, ii] * fluxtilde1
        end

        # y direction
        for jj in eachnode(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
            # pull the contravariant vectors and compute the average
            Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                                   i, jj, element)
            # average mapping terms in second coordinate direction,
            # used as normal vector in the flux computation
            Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
            # compute the contravariant nonconservative flux in the direction of the
            # averaged contravariant vector
            fluxtilde2 = nonconservative_flux(u_node, u_node_jj, Ja2_avg,
                                              equations)
            integral_contribution = integral_contribution +
                                    derivative_split[j, jj] * fluxtilde2
        end

        # The factor 0.5 cancels the factor 2 in the flux differencing form
        multiply_add_to_node_vars!(du, alpha * 0.5f0, integral_contribution, equations,
                                   dg, i, j, element)
    end

    return nothing
end

@inline function flux_differencing_kernel!(du, u,
                                           element,
                                           mesh::Union{StructuredMesh{2},
                                                       StructuredMeshView{2},
                                                       UnstructuredMesh2D, P4estMesh{2},
                                                       T8codeMesh{2}},
                                           have_nonconservative_terms::True,
                                           combine_conservative_and_nonconservative_fluxes::True,
                                           equations,
                                           volume_flux, dg::DGSEM, cache, alpha = true)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements

    # Calculate volume integral in one element
    for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)

        # pull the contravariant vectors in each coordinate direction
        Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, element)
        Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.

        # x direction
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
            # pull the contravariant vectors and compute the average
            Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                                   ii, j, element)
            # average mapping terms in first coordinate direction,
            # used as normal vector in the flux computation
            Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
            # compute the contravariant sharp flux in the direction of the
            # averaged contravariant vector
            fluxtilde1_left, fluxtilde1_right = volume_flux(u_node, u_node_ii, Ja1_avg,
                                                            equations)
            multiply_add_to_node_vars!(du, alpha * derivative_split[i, ii],
                                       fluxtilde1_left,
                                       equations, dg, i, j, element)
            multiply_add_to_node_vars!(du, alpha * derivative_split[ii, i],
                                       fluxtilde1_right,
                                       equations, dg, ii, j, element)
        end

        # y direction
        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
            # pull the contravariant vectors and compute the average
            Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                                   i, jj, element)
            # average mapping terms in second coordinate direction,
            # used as normal vector in the flux computation
            Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
            # compute the contravariant sharp flux in the direction of the
            # averaged contravariant vector
            fluxtilde2_left, fluxtilde2_right = volume_flux(u_node, u_node_jj, Ja2_avg,
                                                            equations)
            multiply_add_to_node_vars!(du, alpha * derivative_split[j, jj],
                                       fluxtilde2_left,
                                       equations, dg, i, j, element)
            multiply_add_to_node_vars!(du, alpha * derivative_split[jj, j],
                                       fluxtilde2_right,
                                       equations, dg, i, jj, element)
        end
    end

    return nothing
end

@inline function calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u,
                              mesh::Union{StructuredMesh{2}, StructuredMeshView{2},
                                          UnstructuredMesh2D,
                                          P4estMesh{2}, T8codeMesh{2}},
                              have_nonconservative_terms::False, equations,
                              volume_flux_fv, dg::DGSEM, element, cache)
    @unpack normal_vectors_1, normal_vectors_2 = cache.normal_vectors

    for j in eachnode(dg), i in 2:nnodes(dg)
        u_ll = get_node_vars(u, equations, dg, i - 1, j, element)
        u_rr = get_node_vars(u, equations, dg, i, j, element)

        # Fetch precomputed freestream-preserving normal vector
        normal_direction = get_normal_vector(normal_vectors_1, i, j, element)

        # Compute the contravariant flux
        contravariant_flux = volume_flux_fv(u_ll, u_rr, normal_direction, equations)

        set_node_vars!(fstar1_L, contravariant_flux, equations, dg, i, j)
        set_node_vars!(fstar1_R, contravariant_flux, equations, dg, i, j)
    end

    for j in 2:nnodes(dg), i in eachnode(dg)
        u_ll = get_node_vars(u, equations, dg, i, j - 1, element)
        u_rr = get_node_vars(u, equations, dg, i, j, element)

        # Fetch precomputed freestream-preserving normal vector
        normal_direction = get_normal_vector(normal_vectors_2, i, j, element)

        # Compute the contravariant flux by taking the scalar product of the
        # normal vector and the flux vector
        contravariant_flux = volume_flux_fv(u_ll, u_rr, normal_direction, equations)

        set_node_vars!(fstar2_L, contravariant_flux, equations, dg, i, j)
        set_node_vars!(fstar2_R, contravariant_flux, equations, dg, i, j)
    end

    return nothing
end

@inline function calcflux_fvO2!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u,
                                mesh::Union{StructuredMesh{2}, StructuredMeshView{2},
                                            UnstructuredMesh2D,
                                            P4estMesh{2}, T8codeMesh{2}},
                                have_nonconservative_terms::False, equations,
                                volume_flux_fv, dg::DGSEM, element, cache,
                                x_interfaces, reconstruction_mode, slope_limiter)
    @unpack normal_vectors_1, normal_vectors_2 = cache.normal_vectors

    # We compute FV02 fluxes at the (nnodes(dg) - 1) subcell boundaries
    # See `calcflux_fvO2!` in solvers/dgsem_tree/dg_1d.jl for a schematic

    # The left subcell node values are labelled `_ll` (left-left) and `_lr` (left-right), while
    # the right subcell node values are labelled `_rl` (right-left) and `_rr` (right-right).
    for j in eachnode(dg), i in 2:nnodes(dg)
        ## Obtain unlimited values in primitive variables ##

        # Note: If i - 2 = 0 we do not go to neighbor element, as one would do in a finite volume scheme.
        # Here, we keep it purely cell-local, thus overshoots between elements are not strictly ruled out,
        # **unless** `reconstruction_mode` is set to `reconstruction_O2_inner`
        u_ll = cons2prim(get_node_vars(u, equations, dg, max(1, i - 2), j, element),
                         equations)
        u_lr = cons2prim(get_node_vars(u, equations, dg, i - 1, j, element),
                         equations)
        u_rl = cons2prim(get_node_vars(u, equations, dg, i, j, element),
                         equations)
        # Note: If i + 1 > nnodes(dg) we do not go to neighbor element, as one would do in a finite volume scheme.
        # Here, we keep it purely cell-local, thus overshoots between elements are not strictly ruled out,
        # **unless** `reconstruction_mode` is set to `reconstruction_O2_inner`
        u_rr = cons2prim(get_node_vars(u, equations, dg, min(nnodes(dg), i + 1), j,
                                       element), equations)

        ## Reconstruct values at interfaces with limiting ##
        u_l, u_r = reconstruction_mode(u_ll, u_lr, u_rl, u_rr,
                                       x_interfaces, i,
                                       slope_limiter, dg)

        # Fetch precomputed freestream-preserving normal vector
        normal_direction = get_normal_vector(normal_vectors_1, i, j, element)

        # Compute the contravariant flux by taking the scalar product of the
        # normal vector and the flux vector.
        ## Convert primitive variables back to conservative variables ##
        contravariant_flux = volume_flux_fv(prim2cons(u_l, equations),
                                            prim2cons(u_r, equations),
                                            normal_direction, equations)

        set_node_vars!(fstar1_L, contravariant_flux, equations, dg, i, j)
        set_node_vars!(fstar1_R, contravariant_flux, equations, dg, i, j)
    end

    for j in 2:nnodes(dg), i in eachnode(dg)
        u_ll = cons2prim(get_node_vars(u, equations, dg, i, max(1, j - 2), element),
                         equations)
        u_lr = cons2prim(get_node_vars(u, equations, dg, i, j - 1, element),
                         equations)
        u_rl = cons2prim(get_node_vars(u, equations, dg, i, j, element),
                         equations)
        u_rr = cons2prim(get_node_vars(u, equations, dg, i, min(nnodes(dg), j + 1),
                                       element), equations)

        u_l, u_r = reconstruction_mode(u_ll, u_lr, u_rl, u_rr,
                                       x_interfaces, j,
                                       slope_limiter, dg)

        normal_direction = get_normal_vector(normal_vectors_2, i, j, element)

        contravariant_flux = volume_flux_fv(prim2cons(u_l, equations),
                                            prim2cons(u_r, equations),
                                            normal_direction, equations)

        set_node_vars!(fstar2_L, contravariant_flux, equations, dg, i, j)
        set_node_vars!(fstar2_R, contravariant_flux, equations, dg, i, j)
    end

    return nothing
end

@inline function calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u,
                              mesh::Union{StructuredMesh{2}, StructuredMesh{2},
                                          UnstructuredMesh2D,
                                          P4estMesh{2}, T8codeMesh{2}},
                              have_nonconservative_terms::True, equations,
                              volume_flux_fv, dg::DGSEM, element, cache)
    @unpack normal_vectors_1, normal_vectors_2 = cache.normal_vectors

    volume_flux, nonconservative_flux = volume_flux_fv

    # Fluxes in x-direction
    for j in eachnode(dg), i in 2:nnodes(dg)
        u_ll = get_node_vars(u, equations, dg, i - 1, j, element)
        u_rr = get_node_vars(u, equations, dg, i, j, element)

        # Fetch precomputed freestream-preserving normal vector
        normal_direction = get_normal_vector(normal_vectors_1, i, j, element)

        # Compute the conservative part of the contravariant flux
        ftilde1 = volume_flux(u_ll, u_rr, normal_direction, equations)

        # Compute and add in the nonconservative part
        # Note the factor 0.5 necessary for the nonconservative fluxes based on
        # the interpretation of global SBP operators coupled discontinuously via
        # central fluxes/SATs
        ftilde1_L = ftilde1 +
                    0.5f0 *
                    nonconservative_flux(u_ll, u_rr, normal_direction, equations)
        ftilde1_R = ftilde1 +
                    0.5f0 *
                    nonconservative_flux(u_rr, u_ll, normal_direction, equations)

        set_node_vars!(fstar1_L, ftilde1_L, equations, dg, i, j)
        set_node_vars!(fstar1_R, ftilde1_R, equations, dg, i, j)
    end

    # Fluxes in y-direction
    for j in 2:nnodes(dg), i in eachnode(dg)
        u_ll = get_node_vars(u, equations, dg, i, j - 1, element)
        u_rr = get_node_vars(u, equations, dg, i, j, element)

        # Fetch precomputed freestream-preserving normal vector
        normal_direction = get_normal_vector(normal_vectors_2, i, j, element)

        # Compute the conservative part of the contravariant flux
        ftilde2 = volume_flux(u_ll, u_rr, normal_direction, equations)

        # Compute and add in the nonconservative part
        # Note the factor 0.5 necessary for the nonconservative fluxes based on
        # the interpretation of global SBP operators coupled discontinuously via
        # central fluxes/SATs
        ftilde2_L = ftilde2 +
                    0.5f0 *
                    nonconservative_flux(u_ll, u_rr, normal_direction, equations)
        ftilde2_R = ftilde2 +
                    0.5f0 *
                    nonconservative_flux(u_rr, u_ll, normal_direction, equations)

        set_node_vars!(fstar2_L, ftilde2_L, equations, dg, i, j)
        set_node_vars!(fstar2_R, ftilde2_R, equations, dg, i, j)
    end

    return nothing
end

function calc_interface_flux!(cache, u,
                              mesh::Union{StructuredMesh{2}, StructuredMeshView{2}},
                              have_nonconservative_terms, # can be True/False
                              equations, surface_integral, dg::DG)
    @unpack elements = cache

    @threaded for element in eachelement(dg, cache)
        # Interfaces in negative directions
        # Faster version of "for orientation in (1, 2)"

        # Interfaces in x-direction (`orientation` = 1)
        calc_interface_flux!(elements.surface_flux_values,
                             elements.left_neighbors[1, element],
                             element, 1, u, mesh,
                             have_nonconservative_terms, equations,
                             surface_integral, dg, cache)

        # Interfaces in y-direction (`orientation` = 2)
        calc_interface_flux!(elements.surface_flux_values,
                             elements.left_neighbors[2, element],
                             element, 2, u, mesh,
                             have_nonconservative_terms, equations,
                             surface_integral, dg, cache)
    end

    return nothing
end

@inline function calc_interface_flux!(surface_flux_values, left_element, right_element,
                                      orientation, u,
                                      mesh::Union{StructuredMesh{2},
                                                  StructuredMeshView{2}},
                                      have_nonconservative_terms::False, equations,
                                      surface_integral, dg::DG, cache)
    # This is slow for LSA, but for some reason faster for Euler (see #519)
    if left_element <= 0 # left_element = 0 at boundaries
        return nothing
    end

    @unpack surface_flux = surface_integral
    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    right_direction = 2 * orientation
    left_direction = right_direction - 1

    for i in eachnode(dg)
        if orientation == 1
            u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, left_element)
            u_rr = get_node_vars(u, equations, dg, 1, i, right_element)

            # If the mapping is orientation-reversing, the contravariant vectors' orientation
            # is reversed as well. The normal vector must be oriented in the direction
            # from `left_element` to `right_element`, or the numerical flux will be computed
            # incorrectly (downwind direction).
            sign_jacobian = sign(inverse_jacobian[1, i, right_element])

            # First contravariant vector Ja^1 as SVector
            normal_direction = sign_jacobian *
                               get_contravariant_vector(1, contravariant_vectors,
                                                        1, i, right_element)
        else # orientation == 2
            u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), left_element)
            u_rr = get_node_vars(u, equations, dg, i, 1, right_element)

            # See above
            sign_jacobian = sign(inverse_jacobian[i, 1, right_element])

            # Second contravariant vector Ja^2 as SVector
            normal_direction = sign_jacobian *
                               get_contravariant_vector(2, contravariant_vectors,
                                                        i, 1, right_element)
        end

        # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
        # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
        flux = sign_jacobian * surface_flux(u_ll, u_rr, normal_direction, equations)

        for v in eachvariable(equations)
            surface_flux_values[v, i, right_direction, left_element] = flux[v]
            surface_flux_values[v, i, left_direction, right_element] = flux[v]
        end
    end

    return nothing
end

@inline function calc_interface_flux!(surface_flux_values, left_element, right_element,
                                      orientation, u,
                                      mesh::Union{StructuredMesh{2},
                                                  StructuredMeshView{2}},
                                      have_nonconservative_terms::True, equations,
                                      surface_integral, dg::DG, cache)
    # See comment on `calc_interface_flux!` with `have_nonconservative_terms::False`
    if left_element <= 0 # left_element = 0 at boundaries
        return nothing
    end

    surface_flux, nonconservative_flux = surface_integral.surface_flux
    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    right_direction = 2 * orientation
    left_direction = right_direction - 1

    for i in eachnode(dg)
        if orientation == 1
            u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, left_element)
            u_rr = get_node_vars(u, equations, dg, 1, i, right_element)

            # If the mapping is orientation-reversing, the contravariant vectors' orientation
            # is reversed as well. The normal vector must be oriented in the direction
            # from `left_element` to `right_element`, or the numerical flux will be computed
            # incorrectly (downwind direction).
            sign_jacobian = sign(inverse_jacobian[1, i, right_element])

            # First contravariant vector Ja^1 as SVector
            normal_direction = sign_jacobian *
                               get_contravariant_vector(1, contravariant_vectors,
                                                        1, i, right_element)
        else # orientation == 2
            u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), left_element)
            u_rr = get_node_vars(u, equations, dg, i, 1, right_element)

            # See above
            sign_jacobian = sign(inverse_jacobian[i, 1, right_element])

            # Second contravariant vector Ja^2 as SVector
            normal_direction = sign_jacobian *
                               get_contravariant_vector(2, contravariant_vectors,
                                                        i, 1, right_element)
        end

        # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
        # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
        flux = sign_jacobian * surface_flux(u_ll, u_rr, normal_direction, equations)

        # Compute both nonconservative fluxes
        # Scale with sign_jacobian to ensure that the normal_direction matches that
        # from the flux above
        noncons_left = sign_jacobian *
                       nonconservative_flux(u_ll, u_rr, normal_direction, equations)
        noncons_right = sign_jacobian *
                        nonconservative_flux(u_rr, u_ll, normal_direction, equations)

        for v in eachvariable(equations)
            # Note the factor 0.5 necessary for the nonconservative fluxes based on
            # the interpretation of global SBP operators coupled discontinuously via
            # central fluxes/SATs
            surface_flux_values[v, i, right_direction, left_element] = flux[v] +
                                                                       0.5f0 *
                                                                       noncons_left[v]
            surface_flux_values[v, i, left_direction, right_element] = flux[v] +
                                                                       0.5f0 *
                                                                       noncons_right[v]
        end
    end

    return nothing
end

function calc_boundary_flux!(cache, u, t, boundary_conditions::NamedTuple,
                             mesh::Union{StructuredMesh{2}, StructuredMeshView{2}},
                             equations, surface_integral,
                             dg::DG)
    @unpack surface_flux_values = cache.elements
    linear_indices = LinearIndices(size(mesh))

    for cell_y in axes(mesh, 2)
        # Negative x-direction
        direction = 1
        element = linear_indices[begin, cell_y]

        for j in eachnode(dg)
            calc_boundary_flux_by_direction!(surface_flux_values, u, t, 1,
                                             boundary_conditions[direction],
                                             mesh,
                                             have_nonconservative_terms(equations),
                                             equations, surface_integral, dg,
                                             cache,
                                             direction, (1, j), (j,), element)
        end

        # Positive x-direction
        direction = 2
        element = linear_indices[end, cell_y]

        for j in eachnode(dg)
            calc_boundary_flux_by_direction!(surface_flux_values, u, t, 1,
                                             boundary_conditions[direction],
                                             mesh,
                                             have_nonconservative_terms(equations),
                                             equations, surface_integral, dg,
                                             cache,
                                             direction, (nnodes(dg), j), (j,), element)
        end
    end

    for cell_x in axes(mesh, 1)
        # Negative y-direction
        direction = 3
        element = linear_indices[cell_x, begin]

        for i in eachnode(dg)
            calc_boundary_flux_by_direction!(surface_flux_values, u, t, 2,
                                             boundary_conditions[direction],
                                             mesh,
                                             have_nonconservative_terms(equations),
                                             equations, surface_integral, dg,
                                             cache,
                                             direction, (i, 1), (i,), element)
        end

        # Positive y-direction
        direction = 4
        element = linear_indices[cell_x, end]

        for i in eachnode(dg)
            calc_boundary_flux_by_direction!(surface_flux_values, u, t, 2,
                                             boundary_conditions[direction],
                                             mesh,
                                             have_nonconservative_terms(equations),
                                             equations, surface_integral, dg,
                                             cache,
                                             direction, (i, nnodes(dg)), (i,), element)
        end
    end

    return nothing
end

function apply_jacobian!(du,
                         mesh::Union{StructuredMesh{2}, StructuredMeshView{2},
                                     UnstructuredMesh2D, P4estMesh{2}, P4estMeshView{2},
                                     T8codeMesh{2}},
                         equations, dg::DG, cache)
    @unpack inverse_jacobian = cache.elements

    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            factor = -inverse_jacobian[i, j, element]

            for v in eachvariable(equations)
                du[v, i, j, element] *= factor
            end
        end
    end

    return nothing
end
end # @muladd
