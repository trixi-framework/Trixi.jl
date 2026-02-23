# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file contains some specialized volume integrals that require some indicators already to be defined.

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

function create_cache(mesh, equations,
                      volume_integral::VolumeIntegralEntropyCorrection,
                      dg, cache_containers, uEltype)
    cache_default = create_cache(mesh, equations,
                                 volume_integral.volume_integral_default,
                                 dg, cache_containers, uEltype)
    cache_stabilized = create_cache(mesh, equations,
                                    volume_integral.volume_integral_stabilized,
                                    dg, cache_containers, uEltype)

    resize!(volume_integral.indicator.cache.alpha, nelements(dg, cache_containers))

    return (; cache_default..., cache_stabilized...)
end

# `resize_volume_integral_cache!` is called after mesh adaptation in `reinitialize_containers!`.
# We only need to resize `volume_integral.indicator.cache.alpha`, which stores the blending factors
# for visualization. 
function resize_volume_integral_cache!(cache, mesh,
                                       volume_integral::VolumeIntegralEntropyCorrection,
                                       new_size)
    @unpack volume_integral_default, volume_integral_stabilized = volume_integral
    resize_volume_integral_cache!(cache, mesh, volume_integral_default, new_size)
    resize_volume_integral_cache!(cache, mesh, volume_integral_stabilized, new_size)

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
    element_variables[:indicator_shock_capturing] = volume_integral.indicator_shock_capturing.cache.alpha
    return nothing
end

# `resize_volume_integral_cache!` is called after mesh adaptation in `reinitialize_containers!`.
# For `VolumeIntegralEntropyCorrectionShockCapturingCombined`, we can reuse the `alpha` array from 
# `indicator_shock_capturing`, which is resized by the call to the shock capturing indicator. 
function resize_volume_integral_cache!(cache, mesh,
                                       volume_integral::VolumeIntegralEntropyCorrectionShockCapturingCombined,
                                       new_size)
    @unpack volume_integral_default, volume_integral_stabilized = volume_integral

    resize_volume_integral_cache!(cache, mesh, volume_integral_default, new_size)
    resize_volume_integral_cache!(cache, mesh, volume_integral_stabilized, new_size)

    return nothing
end
end # @muladd
