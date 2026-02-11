# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    NonIdealCompressibleEulerEquations1D(equation_of_state)

The compressible Euler equations
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
    \rho \\ \rho v_1 \\ \rho e_{\text{total}}
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
    \rho v_1 \\ \rho v_1^2 + p \\ (\rho e_{\text{total}} + p) v_1
\end{pmatrix}
=
\begin{pmatrix}
    0 \\ 0 \\ 0
\end{pmatrix}
```
for a gas with pressure ``p`` specified by some equation of state in one space dimension.

Here, ``\rho`` is the density, ``v_1`` the velocity, ``e_{\text{total}}`` the specific total energy,
and the pressure ``p`` is given in terms of specific volume ``V = 1/\rho`` and temperature ``T``
by some user-specified equation of state (EOS)
(see [`pressure(V, T, eos::IdealGas)`](@ref), [`pressure(V, T, eos::VanDerWaals)`](@ref)) as
```math
p = p(V, T)
```

Similarly, the internal energy is specified by `e_internal = energy_internal_specific(V, T, eos)`, see
[`energy_internal_specific(V, T, eos::IdealGas)`](@ref), [`energy_internal_specific(V, T, eos::VanDerWaals)`](@ref).

Note that this implementation also assumes a mass basis, so molar weight is not taken into account when calculating 
specific volume.
"""
struct NonIdealCompressibleEulerEquations1D{EoS <: AbstractEquationOfState} <:
       AbstractNonIdealCompressibleEulerEquations{1, 3}
    equation_of_state::EoS
end

function varnames(::typeof(cons2cons), ::NonIdealCompressibleEulerEquations1D)
    return ("rho", "rho_v1", "rho_e_total")
end

# returns density, velocity, and pressure
@inline function cons2prim(u,
                           equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    rho, rho_v1, rho_e_total = u
    V, v1, T = cons2thermo(u, equations)
    return SVector(rho, v1, pressure(V, T, eos))
end

varnames(::typeof(cons2prim), ::NonIdealCompressibleEulerEquations1D) = ("rho",
                                                                         "v1",
                                                                         "p")

"""
    cons2thermo(u, equations::NonIdealCompressibleEulerEquations1D)

Convert conservative variables to specific volume, velocity, and temperature
variables `V, v1, T`. These are referred to as "thermodynamic" variables since
equation of state routines are assumed to be evaluated in terms of `V` and `T`.
"""
@inline function cons2thermo(u, equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    rho, rho_v1, rho_e_total = u

    V = inv(rho)
    v1 = rho_v1 * V
    e_internal = (rho_e_total - 0.5f0 * rho_v1 * v1) * V
    T = temperature(V, e_internal, eos)

    return SVector(V, v1, T)
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer,
                      equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state

    _, rho_v1, rho_e_total = u
    V, v1, T = cons2thermo(u, equations)
    p = pressure(V, T, eos)

    # Ignore orientation since it is always "1" in 1D
    f1 = rho_v1
    f2 = rho_v1 * v1 + p
    f3 = (rho_e_total + p) * v1
    return SVector(f1, f2, f3)
end

"""
    flux_terashima_etal(u_ll, u_rr, orientation::Int,
                        equations::NonIdealCompressibleEulerEquations1D)

Approximately pressure equilibrium preserving with conservation (APEC) flux from

- H. Terashima, N. Ly, M. Ihme (2025)
  Approximately pressure-equilibrium-preserving scheme for fully conservative simulations of
  compressible multi-species and real-fluid interfacial flows
  [DOI: 10.1016/j.jcp.2024.113701](https://doi.org/10.1016/j.jcp.2024.113701)

"""
@inline function flux_terashima_etal(u_ll, u_rr, orientation::Int,
                                     equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    V_ll, v1_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, T_rr = cons2thermo(u_rr, equations)

    rho_ll = u_ll[1]
    rho_rr = u_rr[1]
    rho_e_ll = energy_internal(u_ll, equations)
    rho_e_rr = energy_internal(u_rr, equations)
    p_ll = pressure(V_ll, T_ll, eos)
    p_rr = pressure(V_rr, T_rr, eos)

    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    rho_e_avg = 0.5f0 * (rho_e_ll + rho_e_rr)
    p_v1_avg = 0.5f0 * (p_ll * v1_rr + p_rr * v1_ll)

    # chain rule from Terashima
    # Note that `drho_e_internal_drho_p`, i.e., the derivative of the
    # internal energy density with respect to the density at
    # constant pressure is zero for an ideal gas EOS. Thus,
    # the following mean value reduces to
    #   rho_e_internal_corrected_v1_avg = rho_e_internal_corrected_avg * v1_avg
    # for an ideal gas EOS.
    drho_e_internal_drho_p_ll = drho_e_internal_drho_at_const_p(V_ll, T_ll, eos)
    drho_e_internal_drho_p_rr = drho_e_internal_drho_at_const_p(V_rr, T_rr, eos)
    rho_e_internal_corrected_v1_avg = (rho_e_avg -
                                       0.25f0 *
                                       (drho_e_internal_drho_p_rr -
                                        drho_e_internal_drho_p_ll) *
                                       (rho_rr - rho_ll)) * v1_avg

    # Ignore orientation since it is always "1" in 1D
    f_rho = rho_avg * v1_avg
    f_rho_v1 = rho_avg * v1_avg * v1_avg + p_avg
    # Note that the additional "average" is a product and not v1_avg
    f_rho_e_total = rho_e_internal_corrected_v1_avg +
                    rho_avg * 0.5f0 * (v1_ll * v1_rr) * v1_avg + p_v1_avg

    return SVector(f_rho, f_rho_v1, f_rho_e_total)
end

"""
    flux_central_terashima_etal(u_ll, u_rr, orientation::Int,
                                equations::NonIdealCompressibleEulerEquations1D)

A version of the central flux which uses the pressure equilibrium preserving with conservation
(APEC) internal energy correction of
"Approximately pressure-equilibrium-preserving scheme for fully conservative
simulations of compressible multi-species and real-fluid interfacial flows"
by Terashima, Ly, Ihme (2025). <https://doi.org/10.1016/j.jcp.2024.11370>
"""
@inline function flux_central_terashima_etal(u_ll, u_rr, orientation::Int,
                                             equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    V_ll, v1_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, T_rr = cons2thermo(u_rr, equations)

    rho_ll, rho_v1_ll, _ = u_ll
    rho_rr, rho_v1_rr, _ = u_rr
    rho_e_ll = energy_internal(u_ll, equations)
    rho_e_rr = energy_internal(u_rr, equations)
    p_ll = pressure(V_ll, T_ll, eos)
    p_rr = pressure(V_rr, T_rr, eos)

    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    rho_e_avg = 0.5f0 * (rho_e_ll + rho_e_rr)

    # chain rule from Terashima
    # Note that `drho_e_internal_drho_p`, i.e., the derivative of the
    # internal energy density with respect to the density at
    # constant pressure is zero for an ideal gas EOS. Thus,
    # the following mean value reduces to
    #   rho_e_internal_corrected_v1_avg = rho_e_avg * v1_avg
    # for an ideal gas EOS.
    drho_e_internal_drho_p_ll = drho_e_internal_drho_at_const_p(V_ll, T_ll, eos)
    drho_e_internal_drho_p_rr = drho_e_internal_drho_at_const_p(V_rr, T_rr, eos)
    rho_e_internal_corrected_v1_avg = (rho_e_avg -
                                       0.25f0 *
                                       (drho_e_internal_drho_p_rr -
                                        drho_e_internal_drho_p_ll) *
                                       (rho_rr - rho_ll)) *
                                      v1_avg

    # Ignore orientation since it is always "1" in 1D
    f_rho = 0.5f0 * (rho_v1_ll + rho_v1_rr)
    f_rho_v1 = 0.5f0 * (rho_v1_ll * v1_ll + rho_v1_rr * v1_rr) + p_avg

    # calculate internal energy (with APEC correction) and kinetic energy
    # contributions separately in the energy equation
    e_kinetic_ll = 0.5f0 * v1_ll^2
    e_kinetic_rr = 0.5f0 * v1_rr^2
    f_rho_e_total = rho_e_internal_corrected_v1_avg +
                    0.5f0 * (rho_v1_ll * e_kinetic_ll + rho_v1_rr * e_kinetic_rr) +
                    0.5f0 * (p_ll * v1_ll + p_rr * v1_rr)

    return SVector(f_rho, f_rho_v1, f_rho_e_total)
end

# Calculate estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::NonIdealCompressibleEulerEquations1D)
    V_ll, v1_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, T_rr = cons2thermo(u_rr, equations)

    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)
    λ_min = v1_ll - c_ll
    λ_max = v1_rr + c_rr

    return λ_min, λ_max
end

# Less "cautious", i.e., less overestimating `λ_max` compared to `max_abs_speed_naive`
@inline function max_abs_speed(u_ll, u_rr, orientation::Integer,
                               equations::NonIdealCompressibleEulerEquations1D)
    V_ll, v1_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, T_rr = cons2thermo(u_rr, equations)

    v_mag_ll = abs(v1_ll)
    v_mag_rr = abs(v1_rr)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    return max(v_mag_ll + c_ll, v_mag_rr + c_rr)
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::NonIdealCompressibleEulerEquations1D)
    V_ll, v1_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, T_rr = cons2thermo(u_rr, equations)

    v_mag_ll = abs(v1_ll)
    v_mag_rr = abs(v1_rr)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    return max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::NonIdealCompressibleEulerEquations1D)
    V_ll, v1_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, T_rr = cons2thermo(u_rr, equations)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    λ_min = min(v1_ll - c_ll, v1_rr - c_rr)
    λ_max = max(v1_ll + c_ll, v1_rr + c_rr)

    return λ_min, λ_max
end

@inline function max_abs_speeds(u, equations::NonIdealCompressibleEulerEquations1D)
    V, v1, T = cons2thermo(u, equations)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c = speed_of_sound(V, T, eos)

    return (abs(v1) + c,)
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::NonIdealCompressibleEulerEquations1D)
    V, v1, T = cons2thermo(u, equations)
    eos = equations.equation_of_state
    gibbs = gibbs_free_energy(V, T, eos)
    return inv(T) * SVector(gibbs - 0.5f0 * v1^2, v1, -1)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    V, v1, T = prim
    rho = inv(V)
    rho_v1 = rho * v1
    e_internal = energy_internal_specific(V, T, eos)
    rho_e_total = rho * e_internal + 0.5f0 * rho_v1 * v1
    return SVector(rho, rho_v1, rho_e_total)
end

@inline function velocity(u, orientation_or_normal,
                          equations::NonIdealCompressibleEulerEquations1D)
    return velocity(u, equations)
end

@inline function velocity(u, equations::NonIdealCompressibleEulerEquations1D)
    rho = u[1]
    v1 = u[2] / rho
    return v1
end

@inline function energy_internal(u,
                                 equations::NonIdealCompressibleEulerEquations1D)
    rho, rho_v1, rho_e_total = u
    rho_e_internal = rho_e_total - 0.5f0 * rho_v1^2 / rho
    return rho_e_internal
end
end # @muladd
