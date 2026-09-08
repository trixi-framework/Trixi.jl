# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    HelmholtzPengRobinson{RealT <: Real} <: AbstractHelmholtzEOS

Peng-Robinson specific Helmholtz energy from Klein et al., Appendix E, with
``\alpha = c_{v,0}/R`` and the standard squared temperature-dependent attraction factor
``a(T) = a_0 [1 + \kappa (1 - \sqrt{T/T_c})]^2``,
```math
A = - R T \left(1 + \ln\left((V - b) T^{\alpha}\right)\right)
  - \frac{a(T)}{2\sqrt{2}\, b}
    \ln\left(\frac{V + (1 - \sqrt{2}) b}{V + (1 + \sqrt{2}) b}\right).
```

Fields match [`PengRobinson`](@ref).
"""
struct HelmholtzPengRobinson{RealT <: Real} <: AbstractHelmholtzEOS
    R::RealT
    a0::RealT
    b::RealT
    cv0::RealT
    kappa::RealT
    Tc::RealT
    inv2sqrt2b::RealT
    one_minus_sqrt2_b::RealT
    one_plus_sqrt2_b::RealT
end

"""
    HelmholtzPengRobinson(a0, b, cv0, kappa, Tc, R = 8.31446261815324)

Constructs a [`HelmholtzPengRobinson`](@ref) with the same interface as [`PengRobinson`](@ref).
"""
HelmholtzPengRobinson(a0, b, cv0, kappa, Tc, R = 8.31446261815324) = HelmholtzPengRobinson(PengRobinson(a0,
                                                                                                        b,
                                                                                                        cv0,
                                                                                                        kappa,
                                                                                                        Tc,
                                                                                                        R))

function Base.similar(eos::HelmholtzPengRobinson, ::Type{NewRealT}) where {NewRealT}
    return HelmholtzPengRobinson(convert(NewRealT, eos.a0), convert(NewRealT, eos.b),
                                 convert(NewRealT, eos.cv0),
                                 convert(NewRealT, eos.kappa),
                                 convert(NewRealT, eos.Tc), convert(NewRealT, eos.R))
end

"""
    HelmholtzPengRobinson(eos::PengRobinson)

Constructs a [`HelmholtzPengRobinson`](@ref) from an existing [`PengRobinson`](@ref).
"""
function HelmholtzPengRobinson(eos::PengRobinson)
    return HelmholtzPengRobinson{typeof(eos.a0)}(eos.R, eos.a0, eos.b, eos.cv0,
                                                 eos.kappa, eos.Tc,
                                                 eos.inv2sqrt2b, eos.one_minus_sqrt2_b,
                                                 eos.one_plus_sqrt2_b)
end

"""
    HelmholtzPengRobinson(; RealT = Float64)

By default, the units for the Peng-Robinson parameters are in mass basis
(such as kg / m^3) as opposed to molar basis units (such as kg / mol).

The default parameters are for N2.
"""
HelmholtzPengRobinson(; RealT = Float64) = HelmholtzPengRobinson(PengRobinson(; RealT))

@doc raw"""
    helmholtz(V, T, eos::HelmholtzPengRobinson)

Returns the specific Helmholtz energy ``A(V, T)`` for a Peng-Robinson fluid, Klein et al.,
Appendix E.
"""
function helmholtz(V, T, eos::HelmholtzPengRobinson)
    (; R, b, cv0, inv2sqrt2b, one_minus_sqrt2_b, one_plus_sqrt2_b) = eos
    alpha = cv0 / R
    alpha_T = peng_robinson_a(T, eos)
    density_term = (V + one_plus_sqrt2_b) / (V + one_minus_sqrt2_b)
    return -R * T * (1 + log((V - b) * (T^alpha))) -
           inv2sqrt2b * alpha_T * log(density_term)
end
end # @muladd
