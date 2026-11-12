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

    # `VolumeIntegralAdaptive` for `DGMulti` currently limited to Weak Form & Flux Differencing combi
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
    du_values = similar(cache_FD.solution_container.u_values)

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
    @unpack u_values = cache.solution_container

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

            # Recompute using stabilized volume integral. Note that the calculation of this volume integral requires the calculation of the entropy projection, which is done in `rhs_hyperbolic!` specialized on the `DGMultiFluxDiff` solver type.
            volume_integral_kernel!(du, u, e, mesh,
                                    have_nonconservative_terms, equations,
                                    volume_integral_stabilized, dg, cache)
        end
    end

    return nothing
end
# Entropy correction for `DGMulti` solvers with diagonal-norm SBP operators
# (`DGMultiEntropyCorrectionSBP`, see `types.jl`): the weak form volume integral is blended
# elementwise with the entropy stable algebraic low-order method on the SBP nodes
# (`volume_integral_low_order.jl`) such that the volume integral does not produce entropy,
# as for `DGSEM` in `dgsem/calc_volume_integral.jl`. Since the solution nodes are the volume
# quadrature nodes, the entropy production is computed directly from `du`, and the face
# values of the entropy potential are read directly from the volume nodes (`Fmask`, see
# `surface_integral_reference_element` in `callbacks_step/analysis_dgmulti.jl`). The
# solution is stored as a `StructArray` as for `VolumeIntegralWeakForm`, so the weak form
# kernel can be used without changes.

# this method is used when the indicator is constructed as for entropy correction
# volume integrals
function create_cache(::Type{IndicatorEntropyCorrection},
                      equations::AbstractEquations{NDIMS, NVARS},
                      basis::DGMultiBasis) where {NDIMS, NVARS}
    uEltype = real(basis)

    # thread-local storage of the volume integral of a single element
    volume_integral_values_threaded = [Vector{SVector{NVARS, uEltype}}(undef,
                                                                       nnodes(basis))
                                       for _ in 1:Threads.maxthreadid()]

    # stores the blending coefficients
    alpha = Vector{uEltype}()

    return (; alpha, volume_integral_values_threaded)
end

@inline function indicator_entropy_correction(volume_integral::VolumeIntegralEntropyCorrection)
    return volume_integral.indicator
end

@inline function indicator_entropy_correction(volume_integral::VolumeIntegralEntropyCorrectionShockCapturingCombined)
    return volume_integral.indicator.indicator_entropy_correction
end

# Currently restricted to affine meshes; the low-order method does not preserve free-stream
# solutions on curved meshes.
function create_cache(mesh::DGMultiMesh{NDIMS, <:Affine}, equations,
                      dg::DGMultiEntropyCorrectionSBP, RealT, uEltype) where {NDIMS}
    (; volume_integral_default, volume_integral_stabilized) = dg.volume_integral
    if !(volume_integral_default isa VolumeIntegralWeakForm)
        throw(ArgumentError("Entropy correction for `DGMulti` requires " *
                            "`VolumeIntegralWeakForm` as `volume_integral_default`, got " *
                            "$(nameof(typeof(volume_integral_default)))."))
    end
    if !(volume_integral_stabilized isa VolumeIntegralPureLGLFiniteVolume)
        throw(ArgumentError("Entropy correction for `DGMulti` requires " *
                            "`VolumeIntegralPureLGLFiniteVolume` as " *
                            "`volume_integral_stabilized`, got " *
                            "$(nameof(typeof(volume_integral_stabilized)))."))
    end

    # Reuse the cache of the weak form solver, which also contains the containers used by
    # the generic `rhs_hyperbolic!` of `DGMulti`, e.g., for the surface integral.
    dg_WF = DG(dg.basis, dg.mortar, dg.surface_integral, volume_integral_default)
    cache_WF = create_cache(mesh, equations, dg_WF, RealT, uEltype)

    # sparse operators of the low-order method
    cache_low_order = create_cache(mesh, equations, volume_integral_stabilized, dg,
                                   RealT, uEltype)

    # The blending factors are stored in the indicator and used for visualization. They
    # are sized here since `DGMulti` does not use `resize_volume_integral_cache!`.
    (; alpha) = indicator_entropy_correction(dg.volume_integral).cache
    resize!(alpha, mesh.md.num_elements)
    fill!(alpha, zero(eltype(alpha)))

    return (; cache_WF..., cache_low_order...)
end

# Blends the default (weak form) and the stabilized (low-order) volume integral of one
# element such that the volume integral does not produce entropy, and returns the blending
# factor. `alpha_floor` is a lower bound of the blending factor, e.g., from shock capturing,
# and `atol` the tolerance below which `alpha_floor` is considered inactive.
@inline function entropy_correction_kernel!(du, u, element, mesh::DGMultiMesh,
                                            have_nonconservative_terms::False,
                                            equations, volume_integral,
                                            dg::DGMultiEntropyCorrectionSBP, cache,
                                            alpha_floor, atol)
    (; volume_integral_default, volume_integral_stabilized) = volume_integral
    indicator = indicator_entropy_correction(volume_integral)
    (; scaling) = indicator
    du_default = indicator.cache.volume_integral_values_threaded[Threads.threadid()]

    # run default volume integral
    volume_integral_kernel!(du, u, element, mesh,
                            have_nonconservative_terms, equations,
                            volume_integral_default, dg, cache)

    # Check entropy production of the default volume integral. Both the volume and the
    # surface integral are evaluated on the physical element, since `du` contains the
    # geometric terms and the surface integral uses the scaled normal vectors.
    # Minus sign because of the flipped sign of the volume term in the DG RHS.
    du_local = view(du, :, element)
    u_local = view(u, :, element)
    dS_volume_integral = -entropy_change_reference_element(du_local, u_local,
                                                           mesh, equations, dg, cache)

    # Compute true entropy change given by surface integral of the entropy potential
    dS_true = surface_integral_reference_element(entropy_potential, u, element,
                                                 mesh, equations, dg, cache)

    # This quantity should be ≤ 0 for an entropy stable volume integral, and
    # exactly zero for an entropy conservative volume integral.
    entropy_residual = dS_volume_integral - dS_true

    # if either entropy correction or shock capturing is activated
    if entropy_residual > 0 || alpha_floor > atol
        # Store default result
        for i in each_quad_node(mesh, dg, cache)
            du_default[i] = du[i, element]
        end

        # Reset default volume integral contribution.
        # Note that this assumes that the volume terms are computed first,
        # before any surface terms are added.
        fill!(du_local, zero(eltype(du_local)))

        # Calculate entropy stable volume integral contribution
        volume_integral_kernel!(du, u, element, mesh,
                                have_nonconservative_terms, equations,
                                volume_integral_stabilized, dg, cache)

        dS_volume_integral_stabilized = -entropy_change_reference_element(du_local,
                                                                          u_local,
                                                                          mesh,
                                                                          equations,
                                                                          dg, cache)

        # Calculate difference between high and low order entropy production;
        # this should provide positive entropy dissipation if `entropy_residual > 0`,
        # assuming the stabilized volume integral is entropy stable.
        entropy_dissipation = dS_volume_integral_stabilized - dS_volume_integral

        # Calculate the blending factor as the maximum of the entropy correction and
        # `alpha_floor`. `max(0, entropy_residual)` avoids activating entropy correction
        # if shock capturing is active but there is no entropy violation.
        ratio = regularized_ratio(-max(0, entropy_residual), entropy_dissipation)
        alpha_element = max(alpha_floor, min(1, scaling * ratio))

        # Blend the default volume integral back in
        for i in each_quad_node(mesh, dg, cache)
            du[i, element] = alpha_element * du[i, element] +
                             (1 - alpha_element) * du_default[i]
        end

        return alpha_element
    else
        # No entropy correction needed, i.e., pure default volume integral
        return zero(alpha_floor)
    end
end

function calc_volume_integral!(du, u, mesh::DGMultiMesh{NDIMS, <:Affine},
                               have_nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralEntropyCorrection,
                               dg::DGMultiEntropyCorrectionSBP, cache) where {NDIMS}
    (; alpha) = volume_integral.indicator.cache
    check_axes(alpha, (eachelement(mesh, dg, cache),))

    # interpolate to quadrature points (a copy for SBP), required by the weak form kernel
    rd = dg.basis
    (; u_values) = cache.solution_container
    apply_to_each_field(mul_by!(rd.Vq), u_values, u)

    RealT = eltype(alpha)
    @threaded for element in eachelement(mesh, dg, cache)
        alpha[element] = entropy_correction_kernel!(du, u, element, mesh,
                                                    have_nonconservative_terms,
                                                    equations, volume_integral, dg,
                                                    cache, zero(RealT), zero(RealT))
    end

    return nothing
end

function calc_volume_integral!(du, u, mesh::DGMultiMesh{NDIMS, <:Affine},
                               have_nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralEntropyCorrectionShockCapturingCombined,
                               dg::DGMultiEntropyCorrectionSBP, cache) where {NDIMS}
    (; indicator_shock_capturing) = volume_integral.indicator

    # Calculate the shock capturing blending factors a priori. Note that we also reuse
    # the `alpha` array of `indicator_shock_capturing` to store the final blending factors
    # for visualization.
    alpha_shock_capturing = @trixi_timeit timer() "blending factors" begin
        indicator_shock_capturing(u, mesh, equations, dg, cache)
    end
    check_axes(alpha_shock_capturing, (eachelement(mesh, dg, cache),))

    # tolerance to determine "active" blending factors
    # For `Float64`, this gives 1.8189894035458565e-12
    # For `Float32`, this gives 1.1920929f-5
    RealT = eltype(alpha_shock_capturing)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))

    # interpolate to quadrature points (a copy for SBP), required by the weak form kernel
    rd = dg.basis
    (; u_values) = cache.solution_container
    apply_to_each_field(mul_by!(rd.Vq), u_values, u)

    @threaded for element in eachelement(mesh, dg, cache)
        alpha_shock_capturing[element] = entropy_correction_kernel!(du, u, element,
                                                                    mesh,
                                                                    have_nonconservative_terms,
                                                                    equations,
                                                                    volume_integral,
                                                                    dg, cache,
                                                                    alpha_shock_capturing[element],
                                                                    atol)
    end

    return nothing
end
end # @muladd
