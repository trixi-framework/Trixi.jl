# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    VanDerWaals{RealT <: Real} <: AbstractEquationOfState

This defines the van der Waals equation of state
given by the pressure and internal energy relations
```math
p = \frac{\rho R T}{1 - \rho b} - a \rho^2, \quad e = c_v T - a \rho
```
with ``c_v = \frac{R}{\gamma - 1}``. This corresponds to the "simple 
van der Waals" fluid with constant `c_v`, which can be found on p28 of 
<https://math.berkeley.edu/~evans/entropy.and.PDE.pdf>. 

See also "An oscillation free shock-capturing method for compressible van 
der Waals supercritical fluid flows" by Pantano, Saurel, and Schmitt (2017). 
<https://doi.org/10.1016/j.jcp.2017.01.057>
"""
struct VanDerWaals{RealT <: Real} <: AbstractEquationOfState
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

function entropy_specific(V, T, eos::VanDerWaals)
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

# This is not a required interface function, but specializing it 
# if an explicit function is available can improve performance.
function calc_pressure_derivatives(V, T, eos::VanDerWaals)
    (; a, b, R) = eos
    rho = inv(V)
    RT = R * T
    one_minus_b_rho = (1 - b * rho)
    dpdT_V = rho * R / one_minus_b_rho
    dpdrho_T = (RT / one_minus_b_rho + (RT * b * rho) / (one_minus_b_rho^2) -
                2 * a * rho)
    return dpdT_V, -dpdrho_T / V^2
end
end # @muladd
