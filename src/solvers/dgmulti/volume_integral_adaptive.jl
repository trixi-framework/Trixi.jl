# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function create_cache(mesh::DGMultiMesh, equations,
                      dg::DGMulti{NDIMS, ElemType, <:Polynomial,
                                  <:SurfaceIntegralWeakForm,
                                  <:VolumeIntegralAdaptive{<:IndicatorEntropyChange}},
                      RealT, uEltype) where {NDIMS, ElemType}
    # Construct temporary solvers for each sub-integral to reuse their cache allocations.
    # `volume_integral_default` is a weak form integral; `volume_integral_stabilized` is a
    # flux differencing integral.
    dg_wf = DG(dg.basis, dg.mortar, dg.surface_integral,
               dg.volume_integral.volume_integral_default)
    dg_fd = DG(dg.basis, dg.mortar, dg.surface_integral,
               dg.volume_integral.volume_integral_stabilized)

    wf_cache = create_cache(mesh, equations, dg_wf, RealT, uEltype)
    fd_cache = create_cache(mesh, equations, dg_fd, RealT, uEltype)

    rd = dg.basis
    nvars = nvariables(equations)

    # For entropy change difference computation
    du_values = similar(fd_cache.u_values)

    # Thread-local buffer for face interpolation, which is required
    # for computation of entropy potential at interpolated face nodes
    u_face_local_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nfq,), dg)
                             for _ in 1:Threads.maxthreadid()]

    # The FD dxidxhatj ([Vq; Vf] * x) is a superset of the WF one (Vq * x), so it
    # can be shared: the WF kernel only accesses volume-quadrature rows (1:Nq).
    return (; fd_cache...,
            # Weak-form-specific fields required for the default volume integral
            weak_differentiation_matrices = wf_cache.weak_differentiation_matrices,
            lift_scalings = wf_cache.lift_scalings,
            flux_threaded = wf_cache.flux_threaded,
            rotated_flux_threaded = wf_cache.rotated_flux_threaded,
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
