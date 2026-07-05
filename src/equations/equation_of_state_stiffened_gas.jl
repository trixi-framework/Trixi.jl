# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    StiffenedGas{RealT <: Real} <: AbstractEquationOfState

This defines the StiffenedGas equation of state
given by the pressure and internal energy relations
```math
    p = \frac{(e - q)(\gamma - 1)}{V} - \gamma P_{\infty}
```
where ``V = \rho^{-1}``. Heat capacities are constant.

All expressions used here are taken from the following references:

- Ray A. Berry and Richard Saurel and Olivier LeMetayer (2010)
  The discrete equation method (DEM) for fully compressible, 
  two-phase flows in ducts of spatially varying cross-section
  [DOI: 10.1016/j.nucengdes.2010.08.003](https://doi.org/10.1016/j.nucengdes.2010.08.003)

-  Marc O. Delchini and Jean C. Ragusa and Ray A. Berry (2015)
  Entropy-based viscous regularization for the multi-dimensional Euler
  equations in low-Mach and transonic flows
  [DOI: 10.1016/j.compfluid.2015.06.005](https://doi.org/10.1016/j.compfluid.2015.06.005)

"""
struct StiffenedGas{RealT <: Real} <: AbstractEquationOfState
    pInf::RealT
    q::RealT # is the reference specific internal energy
    R::RealT #(takes in J/kg-K), so need to adjust for given specie
    gamma::RealT
    cv0::RealT
end
"""
    StiffenedGas(pInf, q, R, gamma, cv0)

Initializes a Stiffened-Gas equation of state given values for physical constants. 
Here, `R` is the specie gas constant in J/(kg-K) units (computed by dividing univeral gas constant
in appropiate units by the molar mass of the gas specie), pInf, q are constant coefficients
defined by each fluid. gamma is the heat capacity ratio. Default constructor values are
set to liquid water which can be found in the reference below. The naming conventions follows
Section 6 of the following reference:

- Marc O. Delchini and Jean C. Ragusa and Ray A. Berry (2015)
  Entropy-based viscous regularization for the multi-dimensional Euler
  equations in low-Mach and transonic flows
  [DOI: 10.1016/j.compfluid.2015.06.005](https://doi.org/10.1016/j.compfluid.2015.06.005).
"""
function StiffenedGas(; RealT = Float64)
    pInf = 1e9
    q = -1167 * 1e3
    R = 0.08314 / 0.01802
    gamma = 2.35
    cv0 = 1816.0
    return StiffenedGas(pInf, q, R, gamma, cv0)
end
"""
    pressure(V, T, eos::StiffenedGas)

Computes pressure for a Stiffened-Gas gas from specific volume `V` and temperature `T`,
see also [`NonIdealCompressibleEulerEquations1D`](@ref).
"""

function pressure(V, T, eos::StiffenedGas)
    (; pInf, gamma, cv0) = eos
    return T * (gamma - 1) * cv0 / V - pInf
end

@doc raw"""
    energy_internal_specific(V, T, eos::StiffenedGas)

Compute specific internal energy for Stiffened Gas eos
"""
function energy_internal_specific(V, T, eos::StiffenedGas)
    (; q, cv0, pInf) = eos
    return cv0 * T + pInf * V + q
end

@inline function heat_capacity_constant_volume(V, T, eos::StiffenedGas)
    (; cv0) = eos
    return cv0
end

function entropy_specific(V, T, eos::StiffenedGas)
    (; cv0, gamma) = eos
    return cv0 * log((gamma - 1) * cv0 * T * V^(gamma - 1))
end

function speed_of_sound(V, T, eos::StiffenedGas)
    (; gamma, cv0) = eos
    return sqrt((gamma - 1) * gamma * cv0 * T)
end

# This is not a required interface function, but 
# temperature of a StiffenedGas can be computed analytically.
function temperature(V, e_internal, eos::StiffenedGas)
    (; q, pInf, gamma, cv0) = eos
    p = (e_internal - q) * (gamma - 1) / V - gamma * pInf
    return V * (p + pInf) / ((gamma - 1) * cv0)
end

@doc raw"""
    calc_pressure_derivatives(V, T, eos::StiffenedGas)

Compute partial derivative of pressure respect to specific volume, temperature
"""
function calc_pressure_derivatives(V, T, eos::StiffenedGas)
    (; cv0, gamma) = eos
    dpdT_V = (gamma - 1) * cv0 * T / V
    dpdV_T = -(gamma - 1) * cv0 * T / V^2
    return dpdT_V, dpdV_T
end
end # @muladd
