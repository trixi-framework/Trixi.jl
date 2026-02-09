# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    IdealGas{RealT <: Real} <: AbstractEquationOfState

This defines the polytropic ideal gas equation of state
given by the pressure and internal energy relations
```math
p = \rho R T, \quad e = c_v T
```
with ``c_v = \frac{R}{\gamma - 1}``.
"""
struct IdealGas{RealT <: Real} <: AbstractEquationOfState
    gamma::RealT
    R::RealT
    cv::RealT
end

"""
    IdealGas(gamma = 1.4, R = 287)

If not specified, `R` is taken to be the gas constant for air. However, the 
precise value does not matter since eliminating temperature yields non-dimensional
formulas in terms of only `gamma`. 
"""
function IdealGas(gamma = 1.4, R = 287)
    cv = R / (gamma - 1)
    return IdealGas(promote(gamma, R, cv)...)
end

"""
    pressure(V, T, eos::IdealGas)

Computes pressure for an ideal gas from primitive variables (see [`NonIdealCompressibleEulerEquations1D`](@ref))
specific volume `V` and temperature `T`.
"""
function pressure(V, T, eos::IdealGas)
    (; R) = eos
    rho = inv(V)
    p = rho * R * T
    return p
end

"""
    energy_internal(V, T, eos::IdealGas)

Computes internal energy for an ideal gas from specific volume `V` and temperature `T` as
``e = c_v T``.
"""
function energy_internal(V, T, eos::IdealGas)
    (; cv) = eos
    e = cv * T
    return e
end

function entropy_specific(V, T, eos::IdealGas)
    (; cv, R) = eos
    s = cv * log(T) + R * log(V)
    return s
end

function speed_of_sound(V, T, eos::IdealGas)
    (; gamma) = eos
    p = pressure(V, T, eos)
    c2 = gamma * p * V
    return sqrt(c2)
end

# This is not a required interface function, but specializing it 
# if an explicit function is available can improve performance.
# For general EOS, this is calculated via a Newton solve. 
function temperature(V, e, eos::IdealGas)
    (; cv) = eos
    T = e / cv
    return T
end
end # @muladd
