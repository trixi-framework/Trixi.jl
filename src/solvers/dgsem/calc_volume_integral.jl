# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Dimension and meshtype agnostic, i.e., valid for all 1D, 2D, and 3D meshes
function create_cache(mesh, equations,
                      volume_integral::VolumeIntegralFluxDifferencing, dg::DG, uEltype)
    return NamedTuple()
end

# The following `calc_volume_integral!` functions are
# dimension and meshtype agnostic, i.e., valid for all 1D, 2D, and 3D meshes.

function calc_volume_integral!(du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
    @threaded for element in eachelement(dg, cache)
        weak_form_kernel!(du, u, element, mesh,
                          have_nonconservative_terms, equations,
                          dg, cache)
    end

    return nothing
end

function calc_volume_integral!(du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralFluxDifferencing,
                               dg::DGSEM, cache)
    @threaded for element in eachelement(dg, cache)
        flux_differencing_kernel!(du, u, element, mesh,
                                  have_nonconservative_terms, equations,
                                  volume_integral.volume_flux, dg, cache)
    end

    return nothing
end

function calc_volume_integral!(du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingHG,
                               dg::DGSEM, cache)
    @unpack volume_flux_dg, volume_flux_fv, indicator = volume_integral

    # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
    alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations, dg,
                                                               cache)

    # For `Float64`, this gives 1.8189894035458565e-12
    # For `Float32`, this gives 1.1920929f-5
    RealT = eltype(alpha)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))
    @threaded for element in eachelement(dg, cache)
        alpha_element = alpha[element]
        # Clip blending factor for values close to zero (-> pure DG)
        dg_only = isapprox(alpha_element, 0, atol = atol)

        if dg_only
            flux_differencing_kernel!(du, u, element, mesh,
                                      have_nonconservative_terms, equations,
                                      volume_flux_dg, dg, cache)
        else
            # Calculate DG volume integral contribution
            flux_differencing_kernel!(du, u, element, mesh,
                                      have_nonconservative_terms, equations,
                                      volume_flux_dg, dg, cache, 1 - alpha_element)

            # Calculate FV volume integral contribution
            fv_kernel!(du, u, mesh, have_nonconservative_terms, equations,
                       volume_flux_fv, dg, cache, element, alpha_element)
        end
    end

    return nothing
end

function calc_volume_integral!(du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralPureLGLFiniteVolume,
                               dg::DGSEM, cache)
    @unpack volume_flux_fv = volume_integral

    # Calculate LGL FV volume integral
    @threaded for element in eachelement(dg, cache)
        fv_kernel!(du, u, mesh,
                   have_nonconservative_terms, equations,
                   volume_flux_fv, dg, cache, element, true)
    end

    return nothing
end

function calc_volume_integral!(du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralPureLGLFiniteVolumeO2,
                               dg::DGSEM, cache)
    @unpack x_interfaces, volume_flux_fv, reconstruction_mode, slope_limiter = volume_integral

    # Calculate LGL second-order FV volume integral
    @threaded for element in eachelement(dg, cache)
        fvO2_kernel!(du, u, mesh,
                     have_nonconservative_terms, equations,
                     volume_flux_fv, dg, cache, element,
                     x_interfaces, reconstruction_mode, slope_limiter, true)
    end

    return nothing
end
end # @muladd
