# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    HelmholtzVanDerWaals{RealT <: Real} <: AbstractHelmholtzEOS

Van der Waals specific Helmholtz energy from Klein et al., Appendix E, with
``\alpha = c_v/R = 1/(\gamma - 1)``,
```math
A = - R T \left(1 + \ln\left((V - b) T^{\alpha}\right)\right) - \frac{a}{V},
```
equivalent to the density form in Klein et al. with ``\varsigma = 2\alpha = 2/(\gamma - 1)``.

Fields match [`VanDerWaals`](@ref): `a`, `b`, `gamma`, `R`, and precomputed `cv`.
"""
struct HelmholtzVanDerWaals{RealT <: Real} <: AbstractHelmholtzEOS
    a::RealT
    b::RealT
    gamma::RealT
    R::RealT
    cv::RealT
end

"""
    HelmholtzVanDerWaals(; kwargs...)

Constructs a [`HelmholtzVanDerWaals`](@ref) with the same keyword arguments and defaults as
[`VanDerWaals`](@ref).
"""
HelmholtzVanDerWaals(; kwargs...) = HelmholtzVanDerWaals(VanDerWaals(; kwargs...))

"""
    HelmholtzVanDerWaals(eos::VanDerWaals)

Constructs a [`HelmholtzVanDerWaals`](@ref) from an existing [`VanDerWaals`](@ref).
"""
function HelmholtzVanDerWaals(eos::VanDerWaals)
    return HelmholtzVanDerWaals{typeof(eos.a)}(eos.a, eos.b, eos.gamma, eos.R, eos.cv)
end

function Base.similar(eos::HelmholtzVanDerWaals, ::Type{NewRealT}) where {NewRealT}
    return HelmholtzVanDerWaals(; a = convert(NewRealT, eos.a),
                                b = convert(NewRealT, eos.b),
                                gamma = convert(NewRealT, eos.gamma),
                                R = convert(NewRealT, eos.R))
end

@doc raw"""
    helmholtz(V, T, eos::HelmholtzVanDerWaals)

Returns the specific Helmholtz energy ``A(V, T)`` for a van der Waals fluid, Klein et al.,
Appendix E.
"""
function helmholtz(V, T, eos::HelmholtzVanDerWaals)
    (; a, b, R, gamma) = eos
    alpha = inv(gamma - 1)
    return -R * T * (1 + log((V - b) * (T^alpha))) - a / V
end
end # @muladd
