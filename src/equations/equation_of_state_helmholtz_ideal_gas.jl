
@doc raw"""
    HelmholtzIdealGas{RealT <: Real} <: AbstractHelmholtzEOS

Ideal-gas specific Helmholtz energy from Klein et al., Appendix E, equation (E.1).
The specific Helmholtz energy is
```math
A = - R T \left(1 + \ln\left(T^{1/(\gamma-1)} V\right)\right),
```
with ``c_v = R / (\gamma - 1)``.

Fields:
- `gamma`: ratio of specific heats
- `R`: specific gas constant 
"""
struct HelmholtzIdealGas{RealT <: Real} <: AbstractHelmholtzEOS
    gamma::RealT
    R::RealT
end

"""
    HelmholtzIdealGas(gamma = 1.4, R = 287)

Constructs a [`HelmholtzIdealGas`](@ref) with ratio of specific heats 
`gamma` and specific gas constant `R`. If not specified, `R` defaults 
to 287 J/(kg K), a value typical of air (see also [`IdealGas`](@ref)). 
"""
function HelmholtzIdealGas(gamma = 1.4, R = 287)
    return HelmholtzIdealGas(promote(gamma, R)...)
end

@doc raw"""
    helmholtz(V, T, eos::HelmholtzIdealGas)

Returns the specific Helmholtz energy ``A(V, T)`` for an ideal gas, Klein et al.,
equation (E.1), with ``\alpha = 1/(\gamma - 1) = c_v/R``,
```math
A = - R T \left(1 + \ln\left(T^{\alpha} V\right)\right).
```
"""
function helmholtz(V, T, eos::HelmholtzIdealGas)
    alpha = inv(eos.gamma - 1)   # = c_v / R
    return -eos.R * T * (1 + log((T^alpha) * V))
end

@doc raw"""
    temperature(V, e_internal, eos::HelmholtzIdealGas)

This is not a required interface function, but specializing it if an explicit function is
available can improve performance. For general EOS, this is calculated via a Newton solve.

For [`HelmholtzIdealGas`](@ref), ``e_{\text{internal}} = c_v T`` with
``c_v = R / (\gamma - 1)``, so ``T = e_{\text{internal}} / c_v``. The specific volume `V` is
unused.
"""
function temperature(V, e_internal, eos::HelmholtzIdealGas)
    cv = eos.R / (eos.gamma - 1)
    return e_internal / cv
end

@doc raw"""
    speed_of_sound(V, T, eos::HelmholtzIdealGas)

Computes the speed of sound as ``\sqrt{\gamma p V}`` with ``p`` from [`pressure`](@ref) at
`(V, T, eos)`, matching [`IdealGas`](@ref) and equivalent to Klein et al., equation (C.8),
for this EOS. The general [`AbstractHelmholtzEOS`](@ref) implementation evaluates (C.8) from
derivatives of ``A(\rho, T)`` in natural variables ``(\rho, T)``.
"""
function speed_of_sound(V, T, eos::HelmholtzIdealGas)
    (; gamma) = eos
    p = pressure(V, T, eos)
    return sqrt(gamma * p * V)
end
