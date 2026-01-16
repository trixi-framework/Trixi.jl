# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    VanDerWaals{RealT} <: AbstractEquationOfState

This defines the van der Waals equation of state
given by the pressure and internal energy relations
```math
p = \frac{\rho R T}{1 - \rho b} - a \rho^2, \quad e = c_v T - a \rho
```
with ``c_v = \frac{R}{\gamma - 1}``.
"""
struct VanDerWaals{RealT} <: AbstractEquationOfState
    a::RealT
    b::RealT
    gamma::RealT
    R::RealT
    cv::RealT
end

"""
    VanDerWaals(; a = 174.64049524257663, b = 0.001381308696129041,
                gamma = 5 / 3, R = 296.8390795484912)

By default, van der Waals parameters are for N2.
"""
function VanDerWaals(; a = 174.64049524257663, b = 0.001381308696129041,
                     gamma = 5 / 3, R = 296.8390795484912)
    cv = R / (gamma - 1)
    return VanDerWaals(promote(a, b, gamma, R, cv)...)
end

"""
    pressure(V, T, eos::VanDerWaals)

Computes pressure for a van der Waals gas from specific volume `V` and temperature `T`,
see also [`NonIdealCompressibleEulerEquations1D`](@ref).
"""
function pressure(V, T, eos::VanDerWaals)
    (; a, b, R) = eos
    rho = inv(V)
    p = rho * R * T / (1 - rho * b) - a * rho^2
    return p
end

"""
    energy_internal(V, T, eos::VanDerWaals)

Computes internal energy for a van der Waals gas from specific volume `V` and temperature `T` as
``e = c_v T - a \rho``.
"""
function energy_internal(V, T, eos::VanDerWaals)
    (; cv, a) = eos
    rho = inv(V)
    e = cv * T - a * rho
    return e
end

function specific_entropy(V, T, eos::VanDerWaals)
    (; cv, b, R) = eos

    # The specific entropy is defined up to some reference value. The value 
    # s0 = -319.1595051898981 recovers the specific entropy defined in Clapeyron.jl
    s = cv * log(T) + R * log(V - b)
    return s
end

# This formula is taken from (A.26) in the paper "An oscillation free 
# shock-capturing method for compressible van der Waals supercritical 
# fluid flows" by Pantano, Saurel, and Schmitt (2017). 
# https://doi.org/10.1016/j.jcp.2017.01.057
function speed_of_sound(V, T, eos::VanDerWaals)
    (; a, b, gamma) = eos
    rho = inv(V)
    e = energy_internal(V, T, eos)
    c2 = gamma * (gamma - 1) * (e + rho * a) / (1 - rho * b)^2 - 2 * a * rho
    return sqrt(c2)
end

# This is not a required interface function, but specializing it 
# if an explicit function is available can improve performance.
# For general EOS, this is calculated via a Newton solve. 
function temperature(V, e, eos::VanDerWaals)
    (; cv, a) = eos
    rho = inv(V)
    T = (e + a * rho) / cv
    return T
end

end # @muladd
