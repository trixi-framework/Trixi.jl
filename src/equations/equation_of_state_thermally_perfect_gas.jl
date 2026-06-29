@doc raw"""
    ThermallyPerfectGas9PolyFit{R_specific,
    TemperatureBounds,
    Coefficients,
    P_Ref, T_Ref} <: AbstractThermallyPerfectGas

Thermally perfect gas with ideal gas relation
```math
p = \rho R_specific T = \frac{R_specific T}{V}
```
and non-constant, but only temperature-dependent heat capacities ``c_p(T), c_v(T)``
represented by piecewise NASA 9-coefficient polynomials, see
- McBride, Zehe, Gordon (2002).
  NASA Glenn Coefficients for Calculating Thermodynamic Properties of Individual Species.
  [URL](https://ntrs.nasa.gov/citations/20020085330) [PDF](https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf)

For each temperature interval, the dimensionless heat capacity at constant pressure is
```math
\frac{c_p(T)}{R_specific} = a_1 T^{-2} + a_2 T^{-1} + a_3 + a_4 T + a_5 T^2 + a_6 T^3 + a_7 T^4.
```
The corresponding enthalpy and entropy are obtained by integrating `c_p(T)` and
`c_v(T) = c_p(T) - R_specific`.

Fields:
- `R_specific`: specific gas constant, i.e, ``R_\text{universal} / M`` where ``M`` is the molar mass of the gas.
  The molar mass is usually provided with the NASA polynomial data.
- `temperature_bounds`: interval boundaries with length `N + 1`.
  In case the supplied temperature is outside the bounds, the closest interval is used.
- `coefficients`: 9 NASA coefficients per interval, stored column-wise, i.e.,
the dimensions of `coefficients` are `(9, N)` where `N` is the number of temperature intervals.
- `p_ref`: Reference pressure for the entropy calculation, usually 1 bar = 100000 Pa.
- `T_ref`: Reference temperature for the entropy calculation, usually 298.15 K.
Also used as `eos_initial_temperature` for the Newton iteration to find the temperature.
"""
struct ThermallyPerfectGas9PolyFit{R_specific <: Real,
                                   TemperatureBounds <: AbstractVector,
                                   Coefficients <: AbstractMatrix,
                                   P_Ref <: Real, T_Ref <: Real} <:
       AbstractThermallyPerfectGas
    R_specific::R_specific
    temperature_bounds::TemperatureBounds
    coefficients::Coefficients
    p_ref::P_Ref
    T_ref::T_Ref
end

"""
    coefficients_air_9polyfit(temperature_bounds)

NASA 9-coefficient polynomial data for air values correspond to air, see
https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf page 276/284
"""
function coefficients_air_9polyfit(temperature_bounds)
    @assert temperature_bounds==SVector(200.0, 1000.0, 6000.0) "Expected temperature bounds are [200.0, 1000.0, 6000.0]"

    a_cold = [1.009950160e+04; -1.968275610e+02; 5.009155110e+00; -5.761013730e-03;
              1.066859930e-05; -7.940297970e-09; 2.185231910e-12; -1.767967310e+02;
              -3.921504225e+00]
    a_hot = [2.415214430e+05; -1.257874600e+03; 5.144558670e+00; -2.138541790e-04;
             7.065227840e-08; -1.071483490e-11; 6.577800150e-16; 6.462263190e+03;
             -8.147411905e+00]
    a = hcat(a_cold, a_hot)

    return SMatrix{9, 2}(a)
end

"""
    ThermallyPerfectGas9PolyFit(;
                                R_specific = 287.0509010514002,
                                temperature_bounds = SVector(200.0, 1000.0, 6000.0),
                                coefficients = coefficients_air_9polyfit(temperature_bounds),
                                p_ref = 100000.0,
                                T_ref = 298.15)

Construct a [`ThermallyPerfectGas9PolyFit`](@ref) equation of state with NASA 9-coefficient polynomial data.
The default values correspond to air, see
https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf page 276/284
for the coefficient data and 
page 2/10 for the reference temperature (298.15 K) and pressure (1 bar = 100000 Pa).
"""
function ThermallyPerfectGas9PolyFit(;
                                     R_specific = 287.0509010514002,
                                     temperature_bounds = SVector(200.0, 1000.0, 6000.0),
                                     coefficients = coefficients_air_9polyfit(temperature_bounds),
                                     p_ref = 100000.0, T_ref = 298.15)
    @assert size(coefficients, 1)==9 "Current implementation is restricted to NASA 9-coefficient polynomials"

    n_intervals = size(coefficients, 2)
    @assert length(temperature_bounds)==n_intervals + 1 "Temperature bounds do not match the polynomial coefficients"
    @assert issorted(temperature_bounds)

    return ThermallyPerfectGas9PolyFit{typeof(R_specific),
                                       typeof(temperature_bounds),
                                       typeof(coefficients),
                                       typeof(p_ref), typeof(T_ref)}(R_specific,
                                                                     temperature_bounds,
                                                                     coefficients,
                                                                     p_ref, T_ref)
end

# Together with our specialization of `Adapt.adapt_structure`,
# this allows to move semidiscretizations and their components including
# the equations to GPUs and adapt the floating point type, e.g.,
# to `Float32` to improve performance on GPUs.
function Base.similar(eos::ThermallyPerfectGas9PolyFit, ::Type{NewRealT}) where {NewRealT}
    R_specific_conv = convert(NewRealT, eos.R_specific)
    temperature_bounds_conv = convert.(NewRealT, eos.temperature_bounds)
    coefficients_conv = convert.(NewRealT, eos.coefficients)
    p_ref_conv = convert(NewRealT, eos.p_ref)
    T_ref_conv = convert(NewRealT, eos.T_ref)

    return ThermallyPerfectGas9PolyFit{NewRealT,
                                       typeof(temperature_bounds_conv),
                                       typeof(coefficients_conv),
                                       NewRealT, NewRealT}(R_specific_conv,
                                                           temperature_bounds_conv,
                                                           coefficients_conv,
                                                           p_ref_conv,
                                                           T_ref_conv)
end

@inline function temperature_interval(T, eos::ThermallyPerfectGas9PolyFit)
    # Fetch temperature interval index for a given temperature to select the correct polynomial coefficients.
    temp_index = searchsortedlast(eos.temperature_bounds, T)
    # If temperature outside bounds: Use the closest interval, i.e., the first or last interval.
    return clamp(temp_index, 1, length(eos.temperature_bounds) - 1)
end

@doc raw"""
    cp_molar_over_R_universal(T, eos::ThermallyPerfectGas9PolyFit)

Returns `` \frac{c_p(T)}{R_\text{universal}} \: .``
See eq. (1) in https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf
"""
@inline function cp_molar_over_R_universal(T, eos::ThermallyPerfectGas9PolyFit)
    a = eos.coefficients

    idx = temperature_interval(T, eos)

    Tinv = inv(T)
    Tinv2 = Tinv * Tinv
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T

    return a[1, idx] * Tinv2 + a[2, idx] * Tinv + a[3, idx] +
           a[4, idx] * T + a[5, idx] * T2 +
           a[6, idx] * T3 + a[7, idx] * T4
end

@doc raw"""
    h_molar_over_TR_universal(T, eos::ThermallyPerfectGas9PolyFit)

Returns ``\frac{h^\circ(T)}{T \cdot R_\text{universal}} \: .``
See eq. (2) in https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf
"""
@inline function h_molar_over_TR_universal(T, eos::ThermallyPerfectGas9PolyFit)
    a = eos.coefficients

    idx = temperature_interval(T, eos)

    Tinv = inv(T)
    Tinv2 = Tinv * Tinv
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T

    return -a[1, idx] * Tinv2 + a[2, idx] * log(T) * Tinv + a[3, idx] +
           0.5f0 * a[4, idx] * T + (a[5, idx] / 3) * T2 +
           (a[6, idx] / 4) * T3 + (a[7, idx] / 5) * T4 + a[8, idx] * Tinv
end

@doc raw"""
    s_molar_over_R_universal(T, eos::ThermallyPerfectGas9PolyFit)

Returns ``\frac{s^\circ(T)}{R_\text{universal}} \: .``
See eq. (3) in https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf
"""
@inline function s_molar_over_R_universal(T, eos::ThermallyPerfectGas9PolyFit)
    a = eos.coefficients

    idx = temperature_interval(T, eos)

    Tinv = inv(T)
    Tinv2 = Tinv * Tinv
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T

    return -0.5f0 * a[1, idx] * Tinv2 - a[2, idx] * Tinv + a[3, idx] * log(T) +
           a[4, idx] * T + 0.5f0 * a[5, idx] * T2 +
           (a[6, idx] / 3) * T3 + 0.25f0 * a[7, idx] * T4 + a[9, idx]
end

@doc raw"""
    pressure(V, T, eos::ThermallyPerfectGas9PolyFit)

Compute the pressure for a thermally perfect gas with NASA 9-coefficient polynomial data
from the ideal gas relation
```math
p = R_\text{specific} \frac{T}{V}
```
"""
@inline function pressure(V, T, eos::ThermallyPerfectGas9PolyFit)
    # Ideal gas relation
    return eos.R_specific * T / V
end

@inline function heat_capacity_constant_pressure(T, eos::ThermallyPerfectGas9PolyFit)
    # Since we are interested in specific values, we use that
    # cp_specific = cp_molar / M = cp_molar / M * R_universal / R_universal
    #                            = cp_molar / R_universal * R_specific
    return eos.R_specific * cp_molar_over_R_universal(T, eos)
end

@inline function heat_capacity_constant_volume(V, T, eos::ThermallyPerfectGas9PolyFit)
    # cv = cp - R_specific
    return heat_capacity_constant_pressure(V, T, eos) - eos.R_specific
end

@doc raw"""
    energy_internal_specific(V, T, eos::ThermallyPerfectGas9PolyFit)

Computes specific internal energy for a thermally perfect gas with NASA 9-coefficient polynomial data
using the ideal gas relation
``e_{\text{internal}} = h(T) - R_{\text{specific}} T``.
"""
@inline function energy_internal_specific(V, T, eos::ThermallyPerfectGas9PolyFit)
    # Note that h is molar, so to convert to mass-specific we need to divide by the molar mass M.
    # This is achieved by multiplying with R_specific = R_universal / M, since
    # h_molar_over_TR_universal(T, eos) returns h_molar / (T * R_universal).
    return eos.R_specific * T * (h_molar_over_TR_universal(T, eos) - 1)
end

@inline function entropy_specific(V, T, eos::ThermallyPerfectGas9PolyFit)
    @unpack R_specific, p_ref = eos

    # By multiplying with R_specific = R_universal / M, we convert the molar entropy to specific entropy.
    s_ref = R_specific * s_molar_over_R_universal(T, eos) # entropy at reference pressure

    p = pressure(V, T, eos)
    return s_ref + R_specific * log(p_ref / p)
end

# Used in `drho_e_internal_drho_at_const_p` which in turn is used in
# `flux_terashima_etal` and `flux_terashima_etal_central`
@inline function calc_pressure_derivatives(V, T, eos::ThermallyPerfectGas9PolyFit)
    dpdT_V = eos.R_specific / V
    dpdV_T = -eos.R_specific * T / (V^2)
    return dpdT_V, dpdV_T
end

eos_newton_tol(eos::ThermallyPerfectGas9PolyFit) = 1e-8

eos_initial_temperature(V, e_internal, eos::ThermallyPerfectGas9PolyFit) = eos.T_ref # [K]
