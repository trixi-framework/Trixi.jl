# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file contains the storage of DGSEM-specific volume integrals and specialized
# volume integrals that require some indicators already to be defined.

# Subcell-based volume integrals use finite volume fluxes on the subcells. The buffers
# for them are created by `create_cache_subcell_fv`, which dispatches on the mesh type.
function create_cache(mesh, equations,
                      volume_integral::AbstractVolumeIntegralSubcell,
                      dg::DGSEM, cache_containers, uEltype)
    return create_cache_subcell_fv(mesh, equations, dg, cache_containers, uEltype)
end

# Check if subcell limiting should be performed for a given element.
# Always true for pure `VolumeIntegralSubcellLimiting`,
# but not necessarily for `VolumeIntegralAdaptive` with an a-priori indicator.
@inline function perform_subcell_limiting(volume_integral::VolumeIntegralAdaptive{<:IndicatorHennemannGassner},
                                          element)
    return volume_integral.indicator.cache.alpha[element] > 0
end

# Subcell limiting is supported as the `volume_integral_stabilized` of an adaptive
# volume integral with an a-priori indicator, since `perform_subcell_limiting` above
# tells the stage callbacks in which elements subcell limiting was performed.
# See `check_volume_integral_tree` in `solvers/dg.jl`.
function supports_subcell_limiting(volume_integral::VolumeIntegralAdaptive{<:IndicatorHennemannGassner})
    return volume_integral.volume_integral_stabilized isa VolumeIntegralSubcellLimiting
end

const VolumeIntegralAdaptiveEC_WF_DG = VolumeIntegralAdaptive{<:IndicatorEntropyChange,
                                                              <:VolumeIntegralWeakForm,
                                                              <:VolumeIntegralFluxDifferencing}

"""
    VolumeIntegralEntropyCorrection(indicator,
                                    volume_integral_default,
                                    volume_integral_entropy_stable)

Entropy correction volume integral type for DG methods using a convex blending of
a `volume_integral_default` (for example, [`VolumeIntegralWeakForm`](@ref)) and
`volume_integral_entropy_stable` (for example, [`VolumeIntegralPureLGLFiniteVolume`](@ref)
with an entropy stable finite volume flux).

This is intended to be used with [`IndicatorEntropyCorrection`](@ref), which determines the
amount of blending based on the violation of a cell entropy equality by the volume integral.

The parameter `scaling ≥ 1` in [`IndicatorEntropyCorrection`](@ref) scales the DG-FV blending
parameter ``\\alpha``(see the [tutorial on shock-capturing](https://trixi-framework.github.io/TrixiDocumentation/stable/tutorials/shock_capturing/#Shock-capturing-with-flux-differencing))
by a constant, increasing the amount of the subcell FV added in (up to 1, i.e., pure subcell FV).
This can be used to add shock capturing-like behavior. Note though that ``\\alpha`` is computed
here from the entropy defect, **not** using [`IndicatorHennemannGassner`](@ref).

The use of `VolumeIntegralEntropyCorrection` requires either
`entropy_potential(u, orientation, equations)` for TreeMesh, or
`entropy_potential(u, normal_direction, equations)` for other mesh types
to be defined.
"""
const VolumeIntegralEntropyCorrection = VolumeIntegralAdaptive{<:IndicatorEntropyCorrection}

function get_element_variables!(element_variables, u, mesh, equations,
                                volume_integral::VolumeIntegralEntropyCorrection,
                                dg, cache)
    element_variables[:indicator_shock_capturing] = volume_integral.indicator.cache.alpha
    return nothing
end

# The blending factors `alpha` are stored in the indicator (per-instance state), not
# in the `cache`. They are sized here for the elements of the mesh.
function create_cache(mesh, equations,
                      volume_integral::VolumeIntegralEntropyCorrection,
                      dg, cache_containers, uEltype)
    resize!(volume_integral.indicator.cache.alpha, nelements(dg, cache_containers))

    return NamedTuple()
end

# `resize_volume_integral_cache!` is called after mesh adaptation.
# We only need to resize `volume_integral.indicator.cache.alpha`, which stores the blending factors
# for visualization.
function resize_volume_integral_cache!(cache, mesh,
                                       volume_integral::VolumeIntegralEntropyCorrection,
                                       new_size)
    resize!(volume_integral.indicator.cache.alpha, new_size)

    return nothing
end

# `VolumeIntegralEntropyCorrectionShockCapturingCombined` combines the entropy correction
# indicator with a heuristic shock capturing indicator.
const VolumeIntegralEntropyCorrectionShockCapturingCombined = VolumeIntegralAdaptive{<:IndicatorEntropyCorrectionShockCapturingCombined}

function get_element_variables!(element_variables, u, mesh, equations,
                                volume_integral::VolumeIntegralEntropyCorrectionShockCapturingCombined,
                                dg, cache)
    # here, we reuse `indicator_shock_capturing.cache.alpha` to store the indicator variable
    element_variables[:indicator_shock_capturing] = volume_integral.indicator.indicator_shock_capturing.cache.alpha
    return nothing
end
end # @muladd
