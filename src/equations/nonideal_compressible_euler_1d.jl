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
    \rho \\ \rho v_1 \\ \rho e_{total}
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
    \rho v_1 \\ \rho v_1^2 + p \\ (\rho e_{total} + p) v_1
\end{pmatrix}
=
\begin{pmatrix}
    0 \\ 0 \\ 0
\end{pmatrix}
```
for a gas with pressure ``p`` specified by some equation of state in one space dimension.

Here, ``\rho`` is the density, ``v_1`` the velocity, ``e_{total}`` the specific total energy, 
and the pressure ``p`` is given in terms of specific volume ``V = 1/\rho`` and temperature ``T``
by some user-specified equation of state (EOS)
(see [`pressure(V, T, eos::IdealGas)`](@ref), [`pressure(V, T, eos::VanDerWaals)`](@ref)) as
```math
p = p(V, T)
```

Similarly, the internal energy is specified by `e = energy_internal(V, T, eos)`, see
[`energy_internal(V, T, eos::IdealGas)`](@ref), [`energy_internal(V, T, eos::VanDerWaals)`](@ref).

Because of this, the primitive variables are also defined to be `V, v1, T` (instead of 
`rho, v1, p` for `CompressibleEulerEquations1D`). The implementation also assumes 
mass basis unless otherwise specified.     
"""
struct NonIdealCompressibleEulerEquations1D{EoS <: AbstractEquationOfState} <:
       AbstractCompressibleEulerEquations{1, 3}
    equation_of_state::EoS
end

function varnames(::typeof(cons2cons), ::NonIdealCompressibleEulerEquations1D)
    return ("rho", "rho_v1", "rho_e_total")
end
varnames(::typeof(cons2prim), ::NonIdealCompressibleEulerEquations1D) = ("V", "v1", "T")

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer,
                      equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state

    _, rho_v1, rho_e_total = u
    V, v1, T = cons2prim(u, equations)
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
"Approximately pressure-equilibrium-preserving scheme for fully conservative 
simulations of compressible multi-species and real-fluid interfacial flows" 
by Terashima, Ly, Ihme (2025). <https://doi.org/10.1016/j.jcp.2024.11370 1>

"""
@inline function flux_terashima_etal(u_ll, u_rr, orientation::Int,
                                     equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    V_ll, v1_ll, T_ll = cons2prim(u_ll, equations)
    V_rr, v1_rr, T_rr = cons2prim(u_rr, equations)

    rho_ll = u_ll[1]
    rho_rr = u_rr[1]
    rho_e_ll = internal_energy_density(u_ll, equations)
    rho_e_rr = internal_energy_density(u_rr, equations)
    p_ll = pressure(V_ll, T_ll, eos)
    p_rr = pressure(V_rr, T_rr, eos)

    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    rho_e_avg = 0.5f0 * (rho_e_ll + rho_e_rr)
    p_v1_avg = 0.5f0 * (p_ll * v1_rr + p_rr * v1_ll)

    # chain rule from Terashima
    # Note that `drho_e_drho_p`, i.e., the derivative of the
    # internal energy density with respect to the density at
    # constant pressure is zero for an ideal gas EOS. Thus,
    # the following mean value reduces to
    #   rho_e_v1_avg = rho_e_avg * v1_avg
    # for an ideal gas EOS.
    drho_e_drho_p_ll = drho_e_drho_at_const_p(V_ll, T_ll, eos)
    drho_e_drho_p_rr = drho_e_drho_at_const_p(V_rr, T_rr, eos)
    rho_e_v1_avg = (rho_e_avg -
                    0.25f0 * (drho_e_drho_p_rr - drho_e_drho_p_ll) * (rho_rr - rho_ll)) *
                   v1_avg

    # Ignore orientation since it is always "1" in 1D
    f_rho = rho_avg * v1_avg
    f_rho_v1 = rho_avg * v1_avg * v1_avg + p_avg
    # Note that the additional "average" is a product and not v1_avg
    f_rho_E = rho_e_v1_avg + rho_avg * 0.5f0 * (v1_ll * v1_rr) * v1_avg + p_v1_avg

    return SVector(f_rho, f_rho_v1, f_rho_E)
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
    V_ll, v1_ll, T_ll = cons2prim(u_ll, equations)
    V_rr, v1_rr, T_rr = cons2prim(u_rr, equations)

    rho_ll, rho_v1_ll, _ = u_ll
    rho_rr, rho_v1_rr, _ = u_rr
    rho_e_ll = internal_energy_density(u_ll, equations)
    rho_e_rr = internal_energy_density(u_rr, equations)
    p_ll = pressure(V_ll, T_ll, eos)
    p_rr = pressure(V_rr, T_rr, eos)

    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    rho_e_avg = 0.5f0 * (rho_e_ll + rho_e_rr)

    # chain rule from Terashima
    # Note that `drho_e_drho_p`, i.e., the derivative of the
    # internal energy density with respect to the density at
    # constant pressure is zero for an ideal gas EOS. Thus,
    # the following mean value reduces to
    #   rho_e_v1_avg = rho_e_avg * v1_avg
    # for an ideal gas EOS.
    drho_e_drho_p_ll = drho_e_drho_at_const_p(V_ll, T_ll, eos)
    drho_e_drho_p_rr = drho_e_drho_at_const_p(V_rr, T_rr, eos)
    rho_e_v1_avg = (rho_e_avg -
                    0.25f0 * (drho_e_drho_p_rr - drho_e_drho_p_ll) * (rho_rr - rho_ll)) *
                   v1_avg

    # Ignore orientation since it is always "1" in 1D
    f_rho = 0.5f0 * (rho_v1_ll + rho_v1_rr)
    f_rho_v1 = 0.5f0 * (rho_v1_ll * v1_ll + rho_v1_rr * v1_rr) + p_avg

    # calculate internal energy (with APEC correction) and kinetic energy 
    # contributions separately in the energy equation
    ke_ll = 0.5f0 * v1_ll^2
    ke_rr = 0.5f0 * v1_rr^2
    f_rho_E = rho_e_v1_avg +
              0.5f0 * (rho_v1_ll * ke_ll + rho_v1_rr * ke_rr) +
              0.5f0 * (p_ll * v1_ll + p_rr * v1_rr)

    return SVector(f_rho, f_rho_v1, f_rho_E)
end

# Calculate estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::NonIdealCompressibleEulerEquations1D)
    V_ll, v1_ll, T_ll = cons2prim(u_ll, equations)
    V_rr, v1_rr, T_rr = cons2prim(u_rr, equations)

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
    V_ll, v1_ll, T_ll = cons2prim(u_ll, equations)
    V_rr, v1_rr, T_rr = cons2prim(u_rr, equations)

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
    V_ll, v1_ll, T_ll = cons2prim(u_ll, equations)
    V_rr, v1_rr, T_rr = cons2prim(u_rr, equations)

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
    V_ll, v1_ll, T_ll = cons2prim(u_ll, equations)
    V_rr, v1_rr, T_rr = cons2prim(u_rr, equations)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    λ_min = min(v1_ll - c_ll, v1_rr - c_rr)
    λ_max = max(v1_ll + c_ll, v1_rr + c_rr)

    return λ_min, λ_max
end

@inline function max_abs_speeds(u, equations::NonIdealCompressibleEulerEquations1D)
    V, v1, T = cons2prim(u, equations)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c = speed_of_sound(V, T, eos)

    return (abs(v1) + c,)
end

# Convert conservative variables to primitive
@inline function cons2prim(u, equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    rho, rho_v1, rho_e_total = u

    V = inv(rho)
    v1 = rho_v1 * V
    e = (rho_e_total - 0.5f0 * rho_v1 * v1) * V
    T = temperature(V, e, eos)

    return SVector(V, v1, T)
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::NonIdealCompressibleEulerEquations1D)
    V, v1, T = cons2prim(u, equations)
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
    e = energy_internal(V, T, eos)
    rho_e_total = rho * e + 0.5f0 * rho_v1 * v1
    return SVector(rho, rho_v1, rho_e_total)
end

@doc raw"""
    entropy_math(cons, equations::NonIdealCompressibleEulerEquations1D)

Calculate mathematical entropy for a conservative state `cons` as
```math
S = -\rho s
```
where `s` is the specific entropy determined by the equation of state.
"""
@inline function entropy_math(u, equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    V, _, T = cons2prim(u, equations)
    rho = u[1]
    S = -rho * entropy_specific(V, T, eos)
    return S
end

"""
    entropy(cons, equations::NonIdealCompressibleEulerEquations1D)

Default entropy is the mathematical entropy
[`entropy_math(cons, equations::NonIdealCompressibleEulerEquations1D)`](@ref).
"""
@inline function entropy(cons, equations::NonIdealCompressibleEulerEquations1D)
    return entropy_math(cons, equations)
end

@inline function density(u, equations::NonIdealCompressibleEulerEquations1D)
    rho = u[1]
    return rho
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

@inline function pressure(u, equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    V, _, T = cons2prim(u, equations)
    p = pressure(V, T, eos)
    return p
end

@inline function density_pressure(u, equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    rho = u[1]
    V, _, T = cons2prim(u, equations)
    p = pressure(V, T, eos)
    return rho * p
end

@inline function energy_internal(u, equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    V, _, T = cons2prim(u, equations)
    e = energy_internal(V, T, eos)
    return e
end

@inline function internal_energy_density(u,
                                         equations::NonIdealCompressibleEulerEquations1D)
    rho, rho_v1, rho_e_total = u
    rho_e = rho_e_total - 0.5f0 * rho_v1^2 / rho
    return rho_e
end
end # @muladd
