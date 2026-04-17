@doc raw"""
    AbstractHelmholtzEOS

Abstract base type for a thermodynamic description based on the specific Helmholtz energy
``A(V, T)`` as a function of specific volume ``V`` and temperature ``T``.

Subtypes specialize [`helmholtz`](@ref) for arguments `(V, T, eos)`.

Derived quantities follow Klein et al., Appendix~C: (C.3)--(C.5) for entropy, pressure, and
internal energy; (C.6) for specific Gibbs energy; (C.8) for speed of sound. All use specific
volume `V`, temperature `T`, and `eos` as arguments.

Expressions and notation follow:

- R. Klein, B. Sanderse, P. Costa, R. Pecnik, R. Henkes (2026)
  Generalized Tadmor Conditions and Structure-Preserving Numerical Fluxes for the
  Compressible Flow of Real Gases
  [arXiv:2603.15112](https://arxiv.org/abs/2603.15112)
"""
abstract type AbstractHelmholtzEOS <: AbstractEquationOfState end

@doc raw"""
    HelmholtzIdealGas{RealT <: Real} <: AbstractHelmholtzEOS

Ideal-gas specific Helmholtz energy from Klein et al., Appendix~E, equation~(E.1).
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
equation~(E.1), with ``\alpha = 1/(\gamma - 1) = c_v/R``,
```math
A = - R T \left(1 + \ln\left(T^{\alpha} V\right)\right).
```
"""
function helmholtz(V, T, eos::HelmholtzIdealGas)
    alpha = inv(eos.gamma - 1)   # = c_v / R
    return -eos.R * T * (1 + log((T^alpha) * V))
end

@doc raw"""
    pressure(V, T, eos::AbstractHelmholtzEOS)

Computes pressure from specific volume `V` and temperature `T` using the Helmholtz identity
```math
p = -\frac{\partial A}{\partial V}
```
(Klein et al., equation~(C.4)), where ``A`` is given by [`helmholtz`](@ref) at
`(V, T, eos)`.
"""
function pressure(V, T, eos::AbstractHelmholtzEOS)
    return -ForwardDiff.derivative(V_ -> helmholtz(V_, T, eos), V)
end

@doc raw"""
    entropy_specific(V, T, eos::AbstractHelmholtzEOS)

Computes specific entropy from specific volume `V` and temperature `T` using
```math
s = -\frac{\partial A}{\partial T}
```
(Klein et al., equation~(C.3)), where ``A`` is given by [`helmholtz`](@ref) at
`(V, T, eos)`.

This uses the same name and `(V, T, eos)` argument order as [`entropy_specific`](@ref) on
[`AbstractEquationOfState`](@ref).
"""
function entropy_specific(V, T, eos::AbstractHelmholtzEOS)
    return -ForwardDiff.derivative(T_ -> helmholtz(V, T_, eos), T)
end

@doc raw"""
    energy_internal_specific(V, T, eos::AbstractHelmholtzEOS)

Computes specific internal energy from specific volume `V` and temperature `T` using
```math
e = A - T \frac{\partial A}{\partial T}
```
(Klein et al., equation~(C.5)), where ``A`` is given by [`helmholtz`](@ref) at
`(V, T, eos)`.
"""
function energy_internal_specific(V, T, eos::AbstractHelmholtzEOS)
    A = helmholtz(V, T, eos)
    dAdT = ForwardDiff.derivative(T_ -> helmholtz(V, T_, eos), T)
    return A - T * dAdT
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
    gibbs_free_energy(V, T, eos::AbstractHelmholtzEOS)

Computes the specific Gibbs energy using Klein et al., equation~(C.6), expressed in mass
density ``\rho = 1/V`` as ``g = A + \rho \, \partial A / \partial\rho``, which is
equivalent to ``g = A + p V`` when ``A`` is the specific Helmholtz energy as a function of
`(V, T)` with ``p = -\partial A / \partial V``.
"""
function gibbs_free_energy(V, T, eos::AbstractHelmholtzEOS)
    A = helmholtz(V, T, eos)
    p = pressure(V, T, eos)
    return A + p * V
end

@doc raw"""
    speed_of_sound(V, T, eos::AbstractHelmholtzEOS)

Computes the speed of sound using Klein et al., equation~(C.8), with ``A`` expressed in the
natural variables ``(\rho, T)`` and ``\rho = 1/V``:
```math
c^2 = 2\rho \frac{\partial A}{\partial \rho}
    + \rho^2 \frac{\partial^2 A}{\partial \rho^2}
    - \frac{\left(\rho \, \partial^2 A / \partial\rho\,\partial T\right)^2}
           {\partial^2 A / \partial T^2}.
```
"""
function speed_of_sound(V, T, eos::AbstractHelmholtzEOS)
    rho = inv(V)
    A_of_rho(r) = helmholtz(inv(r), T, eos)
    Ar = ForwardDiff.derivative(A_of_rho, rho)
    Arr = ForwardDiff.derivative(r -> ForwardDiff.derivative(A_of_rho, r), rho)
    Att = ForwardDiff.derivative(t_ -> ForwardDiff.derivative(t__ -> helmholtz(inv(rho),
                                                                               t__,
                                                                               eos), t_), T)
    Art = ForwardDiff.derivative(t_ -> begin
                                     A_r(r) = helmholtz(inv(r), t_, eos)
                                     return ForwardDiff.derivative(A_r, rho)
                                 end, T)
    c2 = 2 * rho * Ar + rho^2 * Arr - (rho * Art)^2 / Att
    return sqrt(c2)
end

@doc raw"""
    speed_of_sound(V, T, eos::HelmholtzIdealGas)

Computes the speed of sound as ``\sqrt{\gamma p V}`` with ``p`` from [`pressure`](@ref) at
`(V, T, eos)`, matching [`IdealGas`](@ref) and equivalent to Klein et al., equation~(C.8),
for this EOS. The general [`AbstractHelmholtzEOS`](@ref) implementation evaluates (C.8) from
derivatives of ``A(\rho, T)`` in natural variables ``(\rho, T)``.
"""
function speed_of_sound(V, T, eos::HelmholtzIdealGas)
    (; gamma) = eos
    p = pressure(V, T, eos)
    return sqrt(gamma * p * V)
end
