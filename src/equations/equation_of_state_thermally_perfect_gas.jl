@doc raw"""
    ThermallyPerfectGas{RealT <: Real, N} <: AbstractEquationOfState

Thermally perfect ideal gas equation of state with pressure
```math
p = \rho R T = \frac{R T}{V}
```
and temperature-dependent heat capacity represented by piecewise NASA 9-coefficient
polynomials.

For each temperature interval, the dimensionless heat capacity is
```math
\frac{c_p(T)}{R} = a_1 T^{-2} + a_2 T^{-1} + a_3 + a_4 T + a_5 T^2 + a_6 T^3 + a_7 T^4.
```
The corresponding enthalpy and entropy are obtained by integrating `c_p(T)` and
`c_v(T) = c_p(T) - R`.

Fields:
- `R`: specific gas constant
- `temperature_bounds`: interval boundaries with length `N + 1`
- `coefficients`: 9 NASA coefficients per interval, stored column-wise
"""
struct ThermallyPerfectGas{RealT <: Real, N} <: AbstractEquationOfState
    R::RealT
    temperature_bounds::SVector{N + 1, RealT}
    coefficients::SMatrix{9, N, RealT, 9 * N}
end

"""
    ThermallyPerfectGas(R, temperature_bounds, coefficients)

Construct a thermally perfect gas EOS from NASA 9-coefficient polynomial data.
`temperature_bounds` must have length `size(coefficients, 2) + 1`.
"""
function ThermallyPerfectGas(R, temperature_bounds::AbstractVector,
                             coefficients::AbstractMatrix)
    @assert size(coefficients, 1) == 9
    n_intervals = size(coefficients, 2)
    @assert length(temperature_bounds) == n_intervals + 1
    @assert issorted(temperature_bounds)

    RealT = promote_type(typeof(R), eltype(temperature_bounds), eltype(coefficients))
    return ThermallyPerfectGas{RealT, n_intervals}(convert(RealT, R),
                                                   SVector{n_intervals + 1, RealT}(temperature_bounds),
                                                   SMatrix{9, n_intervals, RealT}(coefficients))
end

"""
    ThermallyPerfectGas(; R, temperature_bounds, coefficients)

Keyword constructor for convenience.
"""
function ThermallyPerfectGas(; R, temperature_bounds, coefficients)
    return ThermallyPerfectGas(R, temperature_bounds, coefficients)
end

@inline function temperature_interval(T, eos::ThermallyPerfectGas{<:Any, N}) where {N}
    return clamp(searchsortedlast(eos.temperature_bounds, T), 1, N)
end

@inline function nasa_coefficients(T, eos::ThermallyPerfectGas)
    return eos.coefficients[:, temperature_interval(T, eos)]
end

@inline function cp_over_R(T, coeffs)
    Tinv = inv(T)
    Tinv2 = Tinv * Tinv
    T2 = T * T
    T3 = T2 * T
    T4 = T2 * T2
    return coeffs[1] * Tinv2 + coeffs[2] * Tinv + coeffs[3] + coeffs[4] * T +
           coeffs[5] * T2 + coeffs[6] * T3 + coeffs[7] * T4
end

@inline function h_over_RT(T, coeffs)
    Tinv = inv(T)
    Tinv2 = Tinv * Tinv
    T2 = T * T
    T3 = T2 * T
    T4 = T2 * T2
    return -coeffs[1] * Tinv2 + coeffs[2] * log(T) * Tinv + coeffs[3] +
           0.5f0 * coeffs[4] * T + (coeffs[5] / 3) * T2 + (coeffs[6] / 4) * T3 +
           (coeffs[7] / 5) * T4 + coeffs[8] * Tinv
end

@inline function s_over_R(T, V, coeffs)
    Tinv = inv(T)
    Tinv2 = Tinv * Tinv
    T2 = T * T
    T3 = T2 * T
    T4 = T2 * T2
    return -0.5f0 * coeffs[1] * Tinv2 - coeffs[2] * Tinv + (coeffs[3] - 1) * log(T) +
           coeffs[4] * T + 0.5f0 * coeffs[5] * T2 + (coeffs[6] / 3) * T3 +
           0.25f0 * coeffs[7] * T4 + coeffs[9] + log(V)
end

@inline function pressure(V, T, eos::ThermallyPerfectGas)
    return eos.R * T / V
end

@inline function heat_capacity_constant_pressure(T, eos::ThermallyPerfectGas)
    coeffs = nasa_coefficients(T, eos)
    return eos.R * cp_over_R(T, coeffs)
end

@inline function heat_capacity_constant_pressure(V, T, eos::ThermallyPerfectGas)
    return heat_capacity_constant_pressure(T, eos)
end

@inline function heat_capacity_constant_volume(V, T, eos::ThermallyPerfectGas)
    return heat_capacity_constant_pressure(V, T, eos) - eos.R
end

@inline function energy_internal_specific(V, T, eos::ThermallyPerfectGas)
    coeffs = nasa_coefficients(T, eos)
    return eos.R * T * (h_over_RT(T, coeffs) - 1)
end

@inline function entropy_specific(V, T, eos::ThermallyPerfectGas)
    coeffs = nasa_coefficients(T, eos)
    return eos.R * s_over_R(T, V, coeffs)
end

@inline function speed_of_sound(V, T, eos::ThermallyPerfectGas)
    gamma_ratio = gamma(T, eos)
    return sqrt(gamma_ratio * pressure(V, T, eos) * V)
end

"""
    gamma(T, eos::ThermallyPerfectGas)

Temperature-dependent ratio of specific heats `c_p(T) / c_v(T)`.
"""
@inline function gamma(T, eos::ThermallyPerfectGas)
    cp = heat_capacity_constant_pressure(T, eos)
    cv = cp - eos.R
    return cp / cv
end

@inline function calc_pressure_derivatives(V, T, eos::ThermallyPerfectGas)
    dpdT_V = eos.R / V
    dpdV_T = -eos.R * T / (V^2)
    return dpdT_V, dpdV_T
end

@inline function eos_initial_temperature(V, e_internal, eos::ThermallyPerfectGas)
    T_min = first(eos.temperature_bounds)
    T_max = last(eos.temperature_bounds)
    T_mid = 0.5f0 * (T_min + T_max)
    e_mid = energy_internal_specific(V, T_mid, eos)
    cv_mid = heat_capacity_constant_volume(V, T_mid, eos)
    T_guess = T_mid + (e_internal - e_mid) / cv_mid
    return clamp(T_guess, T_min, T_max)
end