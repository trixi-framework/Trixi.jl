# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# `VolumeIntegralEntropyCorrectionShockCapturingCombined` combines the entropy correction 
# indicator with a heuristic shock capturing indicator. 
# We define this here since it needs to be defined after the `indicators.jl` are included 
# in `dgsem.jl`, but before `calc_volume_integral.jl` is included.
const VolumeIntegralEntropyCorrectionShockCapturingCombined = VolumeIntegralAdaptive{<:IndicatorEntropyCorrectionShockCapturingCombined}

function get_element_variables!(element_variables, u, mesh, equations,
                                volume_integral::VolumeIntegralEntropyCorrectionShockCapturingCombined,
                                dg, cache)
    element_variables[:indicator_shock_capturing] = volume_integral.indicator_entropy_correction.cache.alpha
    return nothing
end

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
end # @muladd
