# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    PengRobinson{RealT <: Real} <: AbstractEquationOfState

This defines the Peng-Robinson equation of state
given by the pressure and internal energy relations
```math
p = \frac{R T}{V - b} - \frac{a(T)}{V^2 + 2bV - b^2}, \quad e = c_{v,0} T + K(a(T) - Ta'(T))
```
where ``V = inv(rho)`` and auxiliary expressions for ``a(T)`` and ``K`` are given by 
```math
a(T) = a_0\left(1 + \kappa \left 1 - \sqrt{\frac{T}{T_0}}\right)\right)^2, \quad 
K = \frac{1}{b 2\sqrt{2}} \log\left( \frac{V + (1 - b \sqrt{2})}{V + (1 + b\sqrt{2})}\right).
```
Moreover, ``c_v = c_{v,0} - K T a''(T)``. 

All expressions used here are taken from "An entropy-stable hybrid scheme for simulations 
of transcritical real-fluid flows" by Ma, Lv, Ihme (2017).
<https://doi.org/10.1016/j.jcp.2017.03.022>

See also "Towards a fully well-balanced and entropy-stable scheme for the Euler equations with 
gravity: preserving isentropic steady solutions" by Berthon, Michel-Dansac, and Thomann (2024). 
<https://doi.org/10.1016/j.compfluid.2025.106853>

"""
struct PengRobinson{RealT <: Real} <: AbstractEquationOfState
    R::RealT
    a0::RealT
    b::RealT
    cv0::RealT
    kappa::RealT
    T0::RealT
    inv2sqrt2b::RealT
    one_minus_sqrt2_b::RealT
    one_plus_sqrt2_b::RealT
end

"""
    PengRobinson(a0, b, cv0, kappa, T0, R = 8.31446261815324)

Initializes a Peng-Robinson equation of state given values for physical constants. 
Here, ``R`` is the universal gas constant, and the constants ``a0, b, cv0, kappa, T0`` 
follow the naming conventions in Section 2.2 of "Towards a fully well-balanced and 
entropy-stable scheme for the Euler equations with gravity: General equations of 
state" by Michel-Dansac and Thomann (2025). 

<https://doi.org/10.1016/j.compfluid.2025.106853>
"""
function PengRobinson(a0, b, cv0, kappa, T0, R = 8.31446261815324)
    inv2sqrt2b = inv(2 * sqrt(2) * b)
    one_minus_sqrt2_b = (1 - sqrt(2)) * b
    one_plus_sqrt2_b = (1 + sqrt(2)) * b
    return PengRobinson{typeof(a0)}(R, a0, b, cv0, kappa, T0,
                                    inv2sqrt2b, one_minus_sqrt2_b, one_plus_sqrt2_b)
end

"""
    PengRobinson(; RealT = Float64)

By default, the Peng-Robinson parameters are in mass basis for N2. 
"""
function PengRobinson(; RealT = Float64)
    Rgas = 8.31446261815324
    molar_mass = 0.02801 * 1000 # kg/m3
    R = Rgas * 1000 / molar_mass
    pc = 3.40e6
    Tc = 126.2
    omega = 0.0372
    cv0 = 743.2
    b = 0.077796 * R * Tc / pc
    a0 = 0.457236 * (R * Tc)^2 / pc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega^2
    return PengRobinson(RealT.((a0, b, cv0, kappa, Tc, R))...)
end

# the default tolerance of 10 * eps() does not converge for most Peng-Robinson examples,
# so we choose a looser tolerance here. Researchers at the US Naval Research Lab noted 
# that they typically just use 8 fixed Newton iterations for Peng-Robinson.
eos_newton_tol(eos::PengRobinson) = 1e-8

"""
    pressure(V, T, eos::PengRobinson)

Computes pressure for a Peng-Robinson gas from specific volume `V` and temperature `T`,
see also [`NonIdealCompressibleEulerEquations1D`](@ref).
"""
function pressure(V, T, eos::PengRobinson)
    (; R, b) = eos
    p = R * T / (V - b) - peng_robinson_a(T, eos) / (V^2 + 2 * b * V - b^2)
    return p
end

"""
    energy_internal(V, T, eos::PengRobinson)

Computes internal energy for a Peng-Robinson gas from specific volume `V` and temperature `T` as
``e = c_{v,0} T + K_1 (a(T) - T a'(T))``. 
"""
function energy_internal(V, T, eos::PengRobinson)
    (; cv0) = eos
    K1 = calc_K1(V, eos)
    e = cv0 * T + K1 * (peng_robinson_a(T, eos) - T * peng_robinson_da(T, eos))
    return e
end

@inline function heat_capacity_constant_volume(V, T, eos::PengRobinson)
    (; cv0) = eos
    K1 = calc_K1(V, eos)
    cv = cv0 - K1 * T * peng_robinson_d2a(T, eos)
    return cv
end

function entropy_specific(V, T, eos::PengRobinson)
    (; cv0, R, b) = eos

    # The specific entropy is defined up to some reference value s0, which is
    # arbitrarily set to zero here.
    K1 = calc_K1(V, eos)
    return cv0 * log(T) + R * log(V - b) - peng_robinson_da(T, eos) * K1
end

function speed_of_sound(V, T, eos::PengRobinson)
    (; cv0, R, b) = eos

    dpdT_V, dpdV_T = calc_pressure_derivatives(V, T, eos)

    # calculate ratio of specific heats
    K1 = calc_K1(V, eos)
    d2aT = peng_robinson_d2a(T, eos)
    cp0 = cv0 + R
    cv = cv0 - K1 * T * d2aT
    cp = cp0 - R - K1 * T * d2aT - T * dpdT_V^2 / dpdV_T
    gamma = cp / cv

    # calculate bulk modulus, which should be positive 
    # for admissible thermodynamic states.
    kappa_T = -inv(V * dpdV_T)
    c2 = gamma * V / kappa_T
    return sqrt(c2)
end

function calc_pressure_derivatives(V, T, eos::PengRobinson)
    (; R, b) = eos
    denom = (V^2 + 2 * b * V - b^2)
    RdivVb = R / (V - b)
    dpdT_V = RdivVb - peng_robinson_da(T, eos) / denom
    dpdV_T = -RdivVb * T / (V - b) *
             (1 -
              2 * peng_robinson_a(T, eos) / (R * T * (V + b) * (denom / (V^2 - b^2))^2))
    return dpdT_V, dpdV_T
end

# The following are auxiliary functions used in calculating the PR EOS
@inline function peng_robinson_a(T, eos::PengRobinson)
    (; a0, kappa, T0) = eos
    return a0 * (1 + kappa * (1 - sqrt(T / T0)))^2
end
@inline peng_robinson_da(T, eos) = ForwardDiff.derivative(T -> peng_robinson_a(T, eos),
                                                          T)
@inline peng_robinson_d2a(T, eos) = ForwardDiff.derivative(T -> peng_robinson_da(T, eos),
                                                           T)

@inline function calc_K1(V, eos::PengRobinson)
    (; inv2sqrt2b, one_minus_sqrt2_b, one_plus_sqrt2_b) = eos
    K1 = inv2sqrt2b * log((V + one_minus_sqrt2_b) / (V + one_plus_sqrt2_b))
    return K1
end
end # @muladd
