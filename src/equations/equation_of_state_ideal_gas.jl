@doc raw"""
    IdealGas{RealT} <: AbstractEquationOfState

This defines the polytropic ideal gas equation of state
given by the pressure and internal energy relations
```math
p = \rho R T, \quad e = c_v T
```
with ``c_v = \frac{R}{\gamma - 1}``.
"""
struct IdealGas{RealT} <: AbstractEquationOfState
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

function specific_entropy(V, T, eos::IdealGas)
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
