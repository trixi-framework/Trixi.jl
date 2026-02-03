# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Dimension and meshtype agnostic, i.e., valid for all 1D, 2D, and 3D meshes
function create_cache(mesh, equations,
                      volume_integral::VolumeIntegralFluxDifferencing,
                      dg::DG, cache_containers, uEltype)
    return NamedTuple()
end

function create_cache(mesh, equations,
                      volume_integral::VolumeIntegralAdaptive,
                      dg::DG, cache_containers, uEltype)
    # This assumes that `volume_integral.volume_integral_default` needs no special cache!
    @assert volume_integral.volume_integral_default isa VolumeIntegralWeakForm

    return create_cache(mesh, equations,
                        volume_integral.volume_integral_stabilized,
                        dg, cache_containers, uEltype)
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
            flux_differencing_kernel!(du, u, element, mesh,
                                      have_nonconservative_terms, equations,
                                      volume_flux_dg, dg, cache)
        else
            # Calculate DG volume integral contribution
            flux_differencing_kernel!(du, u, element, mesh,
                                      have_nonconservative_terms, equations,
                                      volume_flux_dg, dg, cache, 1 - alpha_element)

            # Calculate FV volume integral contribution
            fv_kernel!(du, u, mesh,
                       have_nonconservative_terms, equations,
                       volume_flux_fv, dg, cache, element, alpha_element)
        end
    end

    return nothing
end

function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{1}, StructuredMesh{1},
                                           TreeMesh{2}, StructuredMesh{2}, P4estMesh{2},
                                           UnstructuredMesh2D, T8codeMesh{2},
                                           TreeMesh{3}},
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingRRG,
                               dg::DGSEM, cache)
    @unpack volume_flux_dg, volume_flux_fv, indicator,
    sc_interface_coords, slope_limiter = volume_integral # Second-oder/RG additions

    # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
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
            flux_differencing_kernel!(du, u, element, mesh,
                                      have_nonconservative_terms, equations,
                                      volume_flux_dg, dg, cache)
        else
            # Calculate DG volume integral contribution
            flux_differencing_kernel!(du, u, element, mesh,
                                      have_nonconservative_terms, equations,
                                      volume_flux_dg, dg, cache, 1 - alpha_element)

            # Calculate second-order FV volume integral contribution
            fvO2_kernel!(du, u, mesh,
                         have_nonconservative_terms, equations,
                         volume_flux_fv, dg, cache, element,
                         # `reconstruction_O2_inner` is needed for limiting effect
                         sc_interface_coords, reconstruction_O2_inner, slope_limiter,
                         alpha_element)
        end
    end

    return nothing
end

function calc_volume_integral!(du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralAdaptive{VolumeIntegralWeakForm,
                                                                       VolumeIntegralFD,
                                                                       Indicator},
                               dg::DGSEM,
                               cache) where {
                                             VolumeIntegralFD <:
                                             VolumeIntegralFluxDifferencing,
                                             Indicator <: IndicatorEntropyChange}
    @unpack volume_integral_default, volume_integral_stabilized = volume_integral

    @threaded for element in eachelement(dg, cache)
        # Compute weak form (WF) volume integral
        weak_form_kernel!(du, u, element, mesh,
                          have_nonconservative_terms, equations,
                          dg, cache)

        # Compute entropy production of the WF volume integral.
        # Minus sign because of the flipped sign of the volume term in the DG RHS.
        # No scaling by inverse Jacobian here, as there is no Jacobian multiplication
        # in `integrate_reference_element`.
        dS_WF = -entropy_change_reference_element(du, u, element,
                                                  mesh, equations, dg, cache)

        # Compute true entropy change given by surface integral of the entropy potential
        dS_true = surface_integral(entropy_potential, u, element,
                                   mesh, equations, dg, cache)

        entropy_change = dS_WF - dS_true
        if entropy_change > 0 # Recompute using EC FD volume integral
            # Reset weak form volume integral contribution
            du[.., element] .= zero(eltype(du))

            # Recompute using entropy-conservative volume integral
            flux_differencing_kernel!(du, u, element, mesh,
                                      have_nonconservative_terms, equations,
                                      volume_integral_stabilized.volume_flux,
                                      dg, cache)
        end
    end

    return nothing
end

function calc_volume_integral!(du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralAdaptive{VolumeIntegralWeakForm,
                                                                       VolumeIntegralSC,
                                                                       Indicator},
                               dg::DGSEM,
                               cache) where {VolumeIntegralSC <:
                                             VolumeIntegralShockCapturingHG,
                                             Indicator <: Nothing}# Indicator taken from `VolumeIntegralSC`
    @unpack volume_flux_dg, volume_flux_fv, indicator = volume_integral.volume_integral_stabilized

    # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
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
            weak_form_kernel!(du, u, element, mesh,
                              have_nonconservative_terms, equations,
                              dg, cache)
        else
            # Calculate DG volume integral contribution
            flux_differencing_kernel!(du, u, element, mesh,
                                      have_nonconservative_terms, equations,
                                      volume_flux_dg, dg, cache, 1 - alpha_element)

            # Calculate FV volume integral contribution
            fv_kernel!(du, u, mesh,
                       have_nonconservative_terms, equations,
                       volume_flux_fv, dg, cache, element, alpha_element)
        end
    end

    return nothing
end

function calc_volume_integral!(du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralAdaptive{VolumeIntegralWeakForm,
                                                                       VolumeIntegralSC,
                                                                       Indicator},
                               dg::DGSEM,
                               cache) where {VolumeIntegralSC <:
                                             VolumeIntegralShockCapturingRRG,
                                             Indicator <: Nothing} # Indicator taken from `VolumeIntegralSC`
    @unpack volume_flux_dg, volume_flux_fv, indicator,
    sc_interface_coords, slope_limiter = volume_integral.volume_integral_stabilized

    # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
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
            weak_form_kernel!(du, u, element, mesh,
                              have_nonconservative_terms, equations,
                              dg, cache)
        else
            # Calculate DG volume integral contribution
            flux_differencing_kernel!(du, u, element, mesh,
                                      have_nonconservative_terms, equations,
                                      volume_flux_dg, dg, cache, 1 - alpha_element)

            # Calculate second-order FV volume integral contribution
            fvO2_kernel!(du, u, mesh,
                         have_nonconservative_terms, equations,
                         volume_flux_fv, dg, cache, element,
                         # `reconstruction_O2_inner` is needed for limiting effect
                         sc_interface_coords, reconstruction_O2_inner, slope_limiter,
                         alpha_element)
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

function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{1}, StructuredMesh{1},
                                           TreeMesh{2}, StructuredMesh{2}, P4estMesh{2},
                                           UnstructuredMesh2D, T8codeMesh{2},
                                           TreeMesh{3}},
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralPureLGLFiniteVolumeO2,
                               dg::DGSEM, cache)
    @unpack sc_interface_coords, volume_flux_fv, reconstruction_mode, slope_limiter = volume_integral

    # Calculate LGL second-order FV volume integral
    @threaded for element in eachelement(dg, cache)
        fvO2_kernel!(du, u, mesh,
                     have_nonconservative_terms, equations,
                     volume_flux_fv, dg, cache, element,
                     sc_interface_coords, reconstruction_mode, slope_limiter, true)
    end

    return nothing
end
end # @muladd
