# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function create_cache(mesh::DGMultiMesh, equations,
                      dg::DGMulti{NDIMS, ElemType, <:Polynomial,
                                  <:SurfaceIntegralWeakForm,
                                  <:VolumeIntegralAdaptive{<:IndicatorEntropyChange,
                                                           <:VolumeIntegralWeakForm,
                                                           <:VolumeIntegralFluxDifferencing}},
                      RealT, uEltype) where {NDIMS, ElemType}
    # Construct temporary solvers for each sub-integral to reuse the `create_cache` functions

    # `VolumeIntegralAdaptive` for `DGMulti` currently limited to Weak From & Flux Differencing combi
    dg_WF = DG(dg.basis, dg.mortar, dg.surface_integral,
               dg.volume_integral.volume_integral_default)
    dg_FD = DG(dg.basis, dg.mortar, dg.surface_integral,
               dg.volume_integral.volume_integral_stabilized)

    cache_WF = create_cache(mesh, equations, dg_WF, RealT, uEltype)
    cache_FD = create_cache(mesh, equations, dg_FD, RealT, uEltype)

    # Set up structures required for `IndicatorEntropyChange`
    rd = dg.basis
    nvars = nvariables(equations)

    # Required for entropy change computation (`entropy_change_reference_element`)
    du_values = similar(cache_FD.u_values)

    # Thread-local buffer for face interpolation, which is required
    # for computation of entropy potential at interpolated face nodes
    # (`surface_integral_reference_element`)
    u_face_local_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nfq,), dg)
                             for _ in 1:Threads.maxthreadid()]

    return (; cache_FD...,
            # Weak-form-specific fields for the default volume integral
            weak_differentiation_matrices = cache_WF.weak_differentiation_matrices,
            flux_threaded = cache_WF.flux_threaded,
            rotated_flux_threaded = cache_WF.rotated_flux_threaded, # For non-affine meshes
            # Required for `IndicatorEntropyChange`
            du_values, u_face_local_threaded)
end

# version for affine meshes (currently only supported one for `VolumeIntegralAdaptive`)
function calc_volume_integral!(du, u, mesh::DGMultiMesh,
                               have_nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralAdaptive{<:IndicatorEntropyChange},
                               dg::DGMultiFluxDiff, cache)
    @unpack volume_integral_default, volume_integral_stabilized = volume_integral
    @unpack maximum_entropy_increase = volume_integral.indicator

    # For weak form integral
    @unpack u_values = cache

    # For entropy production computation
    rd = dg.basis
    @unpack du_values = cache

    # interpolate to quadrature points
    apply_to_each_field(mul_by!(rd.Vq), u_values, u) # required for weak form trial

    @threaded for e in eachelement(dg, cache)
        # Try default volume integral first
        volume_integral_kernel!(du, u, e, mesh,
                                have_nonconservative_terms, equations,
                                volume_integral_default, dg, cache)

        # Interpolate `du` to quadrature points after WF integral for entropy production calculation
        du_local = view(du, :, e)
        du_values_local = view(du_values, :, e)
        apply_to_each_field(mul_by!(rd.Vq), du_values_local, du_local) # required for entropy production calculation

        # Compute entropy production of this volume integral
        u_values_local = view(u_values, :, e)
        dS_WF = -entropy_change_reference_element(du_values_local, u_values_local,
                                                  mesh, equations,
                                                  dg, cache)

        dS_true = surface_integral_reference_element(entropy_potential, u, e,
                                                     mesh, equations, dg, cache)

        entropy_change = dS_WF - dS_true
        if entropy_change > maximum_entropy_increase # Recompute using EC FD volume integral
            # Reset default volume integral contribution.
            # Note that this assumes that the volume terms are computed first,
            # before any surface terms are added.
            fill!(du_local, zero(eltype(du_local)))

            # Recompute using stabilized volume integral
            volume_integral_kernel!(du, u, e, mesh,
                                    have_nonconservative_terms, equations,
                                    volume_integral_stabilized, dg, cache)
        end
    end

    return nothing
end
end # @muladd
