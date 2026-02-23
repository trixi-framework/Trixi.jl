# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# The following `volume_integral_kernel!` and `calc_volume_integral!` functions are
# dimension and meshtype agnostic, i.e., valid for all 1D, 2D, and 3D meshes.

@inline function volume_integral_kernel!(du, u, element, meshT,
                                         have_nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralWeakForm,
                                         dg, cache, alpha = true)
    weak_form_kernel!(du, u, element, meshT,
                      have_nonconservative_terms, equations,
                      dg, cache, alpha)

    return nothing
end

@inline function volume_integral_kernel!(du, u, element, meshT,
                                         have_nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralFluxDifferencing,
                                         dg, cache, alpha = true)
    @unpack volume_flux = volume_integral # Volume integral specific data

    flux_differencing_kernel!(du, u, element, meshT,
                              have_nonconservative_terms, equations,
                              volume_flux, dg, cache, alpha)

    return nothing
end

@inline function volume_integral_kernel!(du, u, element, meshT,
                                         have_nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralPureLGLFiniteVolume,
                                         dg::DGSEM, cache, alpha = true)
    @unpack volume_flux_fv = volume_integral # Volume integral specific data

    fv_kernel!(du, u, meshT,
               have_nonconservative_terms, equations,
               volume_flux_fv, dg, cache, element, alpha)

    return nothing
end

@inline function volume_integral_kernel!(du, u, element, meshT,
                                         have_nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralPureLGLFiniteVolumeO2,
                                         dg::DGSEM, cache, alpha = true)
    # Unpack volume integral specific data
    @unpack sc_interface_coords, volume_flux_fv, reconstruction_mode, slope_limiter = volume_integral

    fvO2_kernel!(du, u, meshT,
                 have_nonconservative_terms, equations,
                 volume_flux_fv, dg, cache, element,
                 sc_interface_coords, reconstruction_mode, slope_limiter, alpha)

    return nothing
end

@inline function volume_integral_kernel!(du, u, element, meshT,
                                         have_nonconservative_terms, equations,
                                         volume_integral::VolumeIntegralAdaptive{<:IndicatorEntropyChange},
                                         dg::DGSEM, cache)
    @unpack volume_integral_default, volume_integral_stabilized, indicator = volume_integral
    @unpack maximum_entropy_increase = indicator

    volume_integral_kernel!(du, u, element, meshT,
                            have_nonconservative_terms, equations,
                            volume_integral_default, dg, cache)

    # Compute entropy production of the default volume integral.
    # Minus sign because of the flipped sign of the volume term in the DG RHS.
    # No scaling by inverse Jacobian here, as there is no Jacobian multiplication
    # in `integrate_reference_element`.
    dS_default = -entropy_change_reference_element(du, u, element,
                                                   meshT, equations, dg, cache)

    # Compute true entropy change given by surface integral of the entropy potential
    dS_true = surface_integral_reference_element(entropy_potential, u, element,
                                                 meshT, equations, dg, cache)

    entropy_change = dS_default - dS_true
    if entropy_change > maximum_entropy_increase # Recompute using EC FD volume integral
        # Reset default volume integral contribution.
        # Note that this assumes that the volume terms are computed first,
        # before any surface terms are added.
        du[.., element] .= zero(eltype(du))

        volume_integral_kernel!(du, u, element, meshT,
                                have_nonconservative_terms, equations,
                                volume_integral_stabilized, dg, cache)
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
    kernel!(du, u, typeof(mesh), have_nonconservative_terms, equations,
            volume_integral, dg, cache,
            ndrange = nelements(dg, cache))
    return nothing
end

@kernel function volume_integral_KAkernel!(du, u, meshT,
                                           have_nonconservative_terms, equations,
                                           volume_integral, dg::DGSEM, cache)
    element = @index(Global)
    volume_integral_kernel!(du, u, element, meshT, have_nonconservative_terms,
                            equations, volume_integral, dg, cache)
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
end # @muladd
