# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# The following `volume_integral_kernel!` and `calc_volume_integral!` functions are
# dimension and meshtype agnostic, i.e., valid for all 1D, 2D, and 3D meshes.

@inline function volume_integral_kernel!(du, u, element, MeshT,
                                         have_nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralWeakForm,
                                         dg, cache, alpha = true)
    weak_form_kernel!(du, u, element, MeshT,
                      have_nonconservative_terms, equations,
                      dg, cache, alpha)

    return nothing
end

@inline function volume_integral_kernel!(du, u, element, MeshT,
                                         have_nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralFluxDifferencing,
                                         dg, cache, alpha = true)
    @unpack volume_flux = volume_integral # Volume integral specific data

    flux_differencing_kernel!(du, u, element, MeshT,
                              have_nonconservative_terms, equations,
                              volume_flux, dg, cache, alpha)

    return nothing
end

@inline function volume_integral_kernel!(du, u, element, MeshT,
                                         have_nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralPureLGLFiniteVolume,
                                         dg::DGSEM, cache, alpha = true)
    @unpack volume_flux_fv = volume_integral # Volume integral specific data

    fv_kernel!(du, u, MeshT,
               have_nonconservative_terms, equations,
               volume_flux_fv, dg, cache, element, alpha)

    return nothing
end

@inline function volume_integral_kernel!(du, u, element, MeshT,
                                         have_nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralPureLGLFiniteVolumeO2,
                                         dg::DGSEM, cache, alpha = true)
    # Unpack volume integral specific data
    @unpack (sc_interface_coords, volume_flux_fv, reconstruction_mode, slope_limiter,
    cons2recon, recon2cons) = volume_integral

    fvO2_kernel!(du, u, MeshT,
                 have_nonconservative_terms, equations,
                 volume_flux_fv, dg, cache, element,
                 sc_interface_coords, reconstruction_mode, slope_limiter,
                 cons2recon, recon2cons,
                 alpha)

    return nothing
end

@inline function volume_integral_kernel!(du, u, element, MeshT,
                                         have_nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralAdaptive{<:IndicatorEntropyChange},
                                         dg::DGSEM, cache)
    @unpack volume_integral_default, volume_integral_stabilized, indicator = volume_integral
    @unpack maximum_entropy_increase = indicator

    volume_integral_kernel!(du, u, element, MeshT,
                            have_nonconservative_terms, equations,
                            volume_integral_default, dg, cache)

    # Compute entropy production of the default volume integral.
    # Minus sign because of the flipped sign of the volume term in the DG RHS.
    # No scaling by inverse Jacobian here, as there is no Jacobian multiplication
    # in `integrate_reference_element`.
    dS_default = -entropy_change_reference_element(du, u, element,
                                                   MeshT, equations, dg, cache)

    # Compute true entropy change given by surface integral of the entropy potential
    dS_true = surface_integral_reference_element(entropy_potential, u, element,
                                                 MeshT, equations, dg, cache)

    entropy_change = dS_default - dS_true
    if entropy_change > maximum_entropy_increase # Recompute using EC FD volume integral
        # Reset default volume integral contribution.
        # Note that this assumes that the volume terms are computed first,
        # before any surface terms are added.
        du[.., element] .= zero(eltype(du))

        volume_integral_kernel!(du, u, element, MeshT,
                                have_nonconservative_terms, equations,
                                volume_integral_stabilized, dg, cache)
    end

    return nothing
end

@inline function volume_integral_kernel!(du, u, element, MeshT,
                                         have_nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralEntropyCorrection,
                                         dg::DGSEM, cache)
    @unpack volume_integral_default, volume_integral_stabilized, indicator = volume_integral
    @unpack scaling = indicator
    @unpack alpha = indicator.cache
    du_element_threaded = indicator.cache.volume_integral_values_threaded

    # run default volume integral 
    volume_integral_kernel!(du, u, element, MeshT,
                            have_nonconservative_terms, equations,
                            volume_integral_default, dg, cache)

    # Check entropy production of "high order" volume integral. 
    # 
    # Note that, for `TreeMesh`, `dS_volume_integral` and `dS_true` are calculated
    # on the reference element. For other mesh types, because ``dS_volume_integral`
    # incorporates the scaled contravariant vectors, `dS_true` should 
    # be calculated on the physical element instead.
    #
    # Minus sign because of the flipped sign of the volume term in the DG RHS.
    # No scaling by inverse Jacobian here, as there is no Jacobian multiplication
    # in `integrate_reference_element`.
    dS_volume_integral = -entropy_change_reference_element(du, u, element,
                                                           MeshT, equations,
                                                           dg, cache)

    # Compute true entropy change given by surface integral of the entropy potential
    dS_true = surface_integral_reference_element(entropy_potential, u, element,
                                                 MeshT, equations, dg, cache)

    # This quantity should be ≤ 0 for an entropy stable volume integral, and 
    # exactly zero for an entropy conservative volume integral. 
    entropy_residual = dS_volume_integral - dS_true

    if entropy_residual > 0
        # Store "high order" result
        du_FD_element = du_element_threaded[Threads.threadid()]
        @views du_FD_element .= du[.., element]

        # Reset pure flux-differencing volume integral 
        # Note that this assumes that the volume terms are computed first,
        # before any surface terms are added.
        du[.., element] .= zero(eltype(du))

        # Calculate entropy stable volume integral contribution
        volume_integral_kernel!(du, u, element, MeshT,
                                have_nonconservative_terms, equations,
                                volume_integral_stabilized, dg, cache)

        dS_volume_integral_stabilized = -entropy_change_reference_element(du, u,
                                                                          element,
                                                                          MeshT,
                                                                          equations, dg,
                                                                          cache)

        # Calculate difference between high and low order FV entropy production;
        # this should provide positive entropy dissipation if `entropy_residual > 0`, 
        # assuming the stabilized volume integral is entropy stable.
        entropy_dissipation = dS_volume_integral_stabilized - dS_volume_integral

        # Calculate DG-FV blending factor 
        ratio = regularized_ratio(-entropy_residual, entropy_dissipation)
        alpha_element = min(1, scaling * ratio) # TODO: replacing this with a differentiable version of `min`

        # Save blending coefficient for visualization
        alpha[element] = alpha_element

        # Blend the high order method back in 
        @views du[.., element] .= alpha_element .* du[.., element] .+
                                  (1 - alpha_element) .* du_FD_element
    end

    return nothing
end

function calc_volume_integral!(backend::Nothing, du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral, dg::DGSEM, cache)
    @threaded for element in eachelement(dg, cache)
        volume_integral_kernel!(du, u, element, typeof(mesh),
                                have_nonconservative_terms, equations,
                                volume_integral, dg, cache)
    end

    return nothing
end

function calc_volume_integral!(backend::Backend, du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral, dg::DGSEM, cache)
    nelements(dg, cache) == 0 && return nothing
    kernel! = volume_integral_KAkernel!(backend)
    kernel_cache = kernel_filter_cache(cache)
    kernel!(du, u, typeof(mesh), have_nonconservative_terms, equations,
            volume_integral, dg, kernel_cache,
            ndrange = nelements(dg, cache))
    return nothing
end

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::P4estMesh{3},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing,
                                       dg::DGSEM, cache)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    nodes = eachnode(dg)
    kernel! = _exp_ijk_div_fusedloop_flux_differencing_kernel!(backend)
    _nnodes = nnodes(dg)
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            dg,
            volume_integral, _nnodes,
            derivative_split,
            contravariant_vectors,
            ndrange = (_nnodes, _nnodes, _nnodes, nelements(dg, cache)))
    return nothing
end

@kernel function _exp_ijk_div_fusedloop_flux_differencing_kernel!(du, u, equations,
                                                                  MeshT::Type{<:P4estMesh{3}},
                                                                  have_nonconservative_terms::False,
                                                                  dg::DGSEM,
                                                                  volume_integral,
                                                                  num_nodes,
                                                                  derivative_split,
                                                                  contravariant_vectors,
                                                                  alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    i, j, k, element = @index(Global, NTuple)

    @unpack volume_flux = volume_integral

    # Calculate volume integral in one element
    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    # pull the contravariant vectors in each coordinate direction
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    # All diagonal entries of `derivative_split` are zero. Thus, we can skip
    # the computation of the diagonal terms. In addition, we use the symmetry
    # of the `volume_flux` to save half of the possible two-point flux
    # computations.

    KernelAbstractions.Extras.@unroll for other in min(i, j, k):num_nodes
        if other > i
            u_node_ii = get_node_vars(u, equations, dg, other, j, k, element)

            # pull the contravariant vectors and compute the average
            Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                                   other, j, k, element)
            Ja1_avg = 0.5 * (Ja1_node + Ja1_node_ii)
            # compute the contravariant sharp flux in the direction of the
            # averaged contravariant vector
            fluxtilde1 = volume_flux(u_node, u_node_ii, Ja1_avg, equations)
            multiply_add_to_first_axis_atomic!(du, alpha * derivative_split[i, other],
                                               fluxtilde1,
                                               i, j, k, element)
            multiply_add_to_first_axis_atomic!(du, alpha * derivative_split[other, i],
                                               fluxtilde1,
                                               other, j, k, element)
        end
        if other > j
            u_node_jj = get_node_vars(u, equations, dg, i, other, k, element)
            # pull the contravariant vectors and compute the average
            Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                                   i, other, k, element)
            Ja2_avg = 0.5 * (Ja2_node + Ja2_node_jj)
            # compute the contravariant sharp flux in the direction of the
            # averaged contravariant vector
            fluxtilde2 = volume_flux(u_node, u_node_jj, Ja2_avg, equations)
            multiply_add_to_first_axis_atomic!(du, alpha * derivative_split[j, other],
                                               fluxtilde2,
                                               i, j, k, element)
            multiply_add_to_first_axis_atomic!(du, alpha * derivative_split[other, j],
                                               fluxtilde2,
                                               i, other, k, element)
        end
        if other > k
            u_node_kk = get_node_vars(u, equations, dg, i, j, other, element)
            # pull the contravariant vectors and compute the average
            Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                                   i, j, other, element)
            Ja3_avg = 0.5 * (Ja3_node + Ja3_node_kk)
            # compute the contravariant sharp flux in the direction of the
            # averaged contravariant vector
            fluxtilde3 = volume_flux(u_node, u_node_kk, Ja3_avg, equations)
            multiply_add_to_first_axis_atomic!(du, alpha * derivative_split[k, other],
                                               fluxtilde3,
                                               i, j, k, element)
            multiply_add_to_first_axis_atomic!(du, alpha * derivative_split[other, k],
                                               fluxtilde3,
                                               i, j, other, element)
        end
    end
end

@kernel function volume_integral_KAkernel!(du, u, MeshT,
                                           have_nonconservative_terms, equations,
                                           volume_integral, dg::DGSEM, cache)
    element = @index(Global)
    volume_integral_kernel!(du, u, element, MeshT, have_nonconservative_terms,
                            equations, volume_integral, dg, cache)
end

@inline function calc_volume_integral!(backend::Nothing, du, u, mesh,
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralAdaptive{<:IndicatorHennemannGassner},
                                       dg::DGSEM, cache)
    @unpack volume_integral_default, volume_integral_stabilized, indicator = volume_integral

    # Calculate a-priori stabilization indicator
    alpha = @trixi_timeit timer() "indicator" indicator(u, mesh, equations,
                                                        dg, cache)

    # For `Float64`, this gives 1.8189894035458565e-12
    # For `Float32`, this gives 1.1920929f-5
    RealT = eltype(alpha)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))
    @threaded for element in eachelement(dg, cache)
        alpha_element = alpha[element]
        # Clip blending factor for values close to zero (-> default volume integral)
        default_volume_integral = isapprox(alpha_element, 0, atol = atol)

        if default_volume_integral
            volume_integral_kernel!(du, u, element, typeof(mesh),
                                    have_nonconservative_terms, equations,
                                    volume_integral_default, dg, cache)
        else
            volume_integral_kernel!(du, u, element, typeof(mesh),
                                    have_nonconservative_terms, equations,
                                    volume_integral_stabilized, dg, cache)
        end
    end

    return nothing
end

function calc_volume_integral!(backend::Nothing, du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingHGType,
                               dg::DGSEM, cache)
    @unpack (indicator, volume_integral_default,
    volume_integral_blend_high_order, volume_integral_blend_low_order) = volume_integral

    # Calculate DG-FV blending factors α a-priori for: u_{DG-FV} = u_DG * (1 - α) + u_FV * α
    alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations,
                                                               dg, cache)

    # For `Float64`, this gives 1.8189894035458565e-12
    # For `Float32`, this gives 1.1920929f-5
    RealT = eltype(alpha)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))
    @threaded for element in eachelement(dg, cache)
        alpha_element = alpha[element]
        # Clip blending factor for values close to zero (-> pure DG)
        dg_only = isapprox(alpha_element, 0, atol = atol)

        if dg_only
            volume_integral_kernel!(du, u, element, typeof(mesh),
                                    have_nonconservative_terms, equations,
                                    volume_integral_default, dg, cache)
        else
            # Calculate DG volume integral contribution
            volume_integral_kernel!(du, u, element, typeof(mesh),
                                    have_nonconservative_terms, equations,
                                    volume_integral_blend_high_order, dg, cache,
                                    1 - alpha_element)

            # Calculate FV volume integral contribution
            volume_integral_kernel!(du, u, element, typeof(mesh),
                                    have_nonconservative_terms, equations,
                                    volume_integral_blend_low_order, dg, cache,
                                    alpha_element)
        end
    end

    return nothing
end

function calc_volume_integral!(backend::Nothing, du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralEntropyCorrectionShockCapturingCombined,
                               dg::DGSEM, cache)
    (; volume_integral_default, volume_integral_stabilized, indicator) = volume_integral
    (; indicator_entropy_correction, indicator_shock_capturing) = indicator
    (; scaling) = indicator_entropy_correction
    du_element_threaded = indicator_entropy_correction.cache.volume_integral_values_threaded

    # Calculate DG-FV blending factors α a-priori for: u_{DG-FV} = u_DG * (1 - α) + u_FV * α
    # Note that we also reuse the `alpha_shock_capturing` array to store the indicator values for visualization.
    alpha_shock_capturing = @trixi_timeit timer() "blending factors" indicator_shock_capturing(u,
                                                                                               mesh,
                                                                                               equations,
                                                                                               dg,
                                                                                               cache)

    @threaded for element in eachelement(dg, cache)
        # run default volume integral 
        volume_integral_kernel!(du, u, element, typeof(mesh),
                                have_nonconservative_terms, equations,
                                volume_integral_default, dg, cache)

        # Check entropy production of "high order" volume integral. 
        # 
        # Note that, for `TreeMesh`, both volume and surface integrals are calculated
        # on the reference element. For other mesh types, because the volume integral 
        # incorporates the scaled contravariant vectors, the surface integral should 
        # be calculated on the physical element instead.
        #
        # Minus sign because of the flipped sign of the volume term in the DG RHS.
        # No scaling by inverse Jacobian here, as there is no Jacobian multiplication
        # in `integrate_reference_element`.
        dS_volume_integral = -entropy_change_reference_element(du, u, element,
                                                               typeof(mesh), equations,
                                                               dg, cache)

        # Compute true entropy change given by surface integral of the entropy potential
        dS_true = surface_integral_reference_element(entropy_potential, u, element,
                                                     typeof(mesh), equations, dg, cache)

        # This quantity should be ≤ 0 for an entropy stable volume integral, and 
        # exactly zero for an entropy conservative volume integral. 
        entropy_residual = dS_volume_integral - dS_true

        if entropy_residual > 0
            # Store "high order" result
            du_FD_element = du_element_threaded[Threads.threadid()]
            @views du_FD_element .= du[.., element]

            # Reset pure flux-differencing volume integral 
            # Note that this assumes that the volume terms are computed first,
            # before any surface terms are added.
            du[.., element] .= zero(eltype(du))

            # Calculate entropy stable volume integral contribution
            volume_integral_kernel!(du, u, element, typeof(mesh),
                                    have_nonconservative_terms, equations,
                                    volume_integral_stabilized, dg, cache)

            dS_volume_integral_stabilized = -entropy_change_reference_element(du, u,
                                                                              element,
                                                                              typeof(mesh),
                                                                              equations,
                                                                              dg,
                                                                              cache)

            # Calculate difference between high and low order FV entropy production;
            # this should provide positive entropy dissipation if `entropy_residual > 0`, 
            # assuming the stabilized volume integral is entropy stable.
            entropy_dissipation = dS_volume_integral_stabilized - dS_volume_integral

            # Calculate DG-FV blending factor as the minimum between the entropy correction 
            # indicator and shock capturing indicator
            # TODO: replacing this with a differentiable version of `min`
            ratio = regularized_ratio(-entropy_residual, entropy_dissipation)
            alpha_element = min(1, max(alpha_shock_capturing[element], scaling * ratio))

            # Save blending coefficient for visualization. Note that we overwrite the data 
            # in `alpha_shock_capturing[element]`. 
            alpha_shock_capturing[element] = alpha_element

            # Blend the high order method back in 
            @views du[.., element] .= alpha_element .* du[.., element] .+
                                      (1 - alpha_element) .* du_FD_element
        end
    end

    return nothing
end
end # @muladd
