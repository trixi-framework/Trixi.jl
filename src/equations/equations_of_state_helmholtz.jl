@doc raw"""
    AbstractHelmholtzEOS

Abstract base type for a thermodynamic description based on the specific Helmholtz energy
``A(V, T)`` as a function of specific volume ``V`` and temperature ``T``.

Subtypes specialize [`helmholtz`](@ref) for arguments `(V, T, eos)`.

Derived quantities follow Klein et al., Appendix C: (C.3)--(C.5) for entropy, pressure, and
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
    pressure(V, T, eos::AbstractHelmholtzEOS)

Computes pressure from specific volume `V` and temperature `T` using the Helmholtz identity
```math
p = -\frac{\partial A}{\partial V}
```
(Klein et al., equation (C.4)), where ``A`` is given by [`helmholtz`](@ref) at
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
(Klein et al., equation (C.3)), where ``A`` is given by [`helmholtz`](@ref) at
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
(Klein et al., equation (C.5)), where ``A`` is given by [`helmholtz`](@ref) at
`(V, T, eos)`.
"""
function energy_internal_specific(V, T, eos::AbstractHelmholtzEOS)
    A = helmholtz(V, T, eos)
    dAdT = ForwardDiff.derivative(T_ -> helmholtz(V, T_, eos), T)
    return A - T * dAdT
end

@doc raw"""
    gibbs_free_energy(V, T, eos::AbstractHelmholtzEOS)

Computes the specific Gibbs energy using Klein et al., equation (C.6), expressed in mass
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

Computes the speed of sound using Klein et al., equation (C.8), with ``A`` expressed in the
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
