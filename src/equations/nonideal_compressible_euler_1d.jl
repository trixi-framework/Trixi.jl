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
    \rho \\ \rho v_1 \\ \rho e_total
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
    \rho v_1 \\ \rho v_1^2 + p \\ (\rho e_total + p) v_1
\end{pmatrix}
=
\begin{pmatrix}
    0 \\ 0 \\ 0
\end{pmatrix}
```
for a gas with pressure ``p`` specified by some equation of state in one space dimension.

Here, ``\rho`` is the density, ``v_1`` the velocity, ``e_total`` the specific total energy, 
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
    e = (rho_e_total - 0.5 * rho_v1 * v1) * V
    T = temperature(V, e, eos)

    return SVector(V, v1, T)
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::NonIdealCompressibleEulerEquations1D)
    V, v1, T = cons2prim(u, equations)
    eos = equations.equation_of_state
    gibbs = gibbs_free_energy(V, T, eos)
    return inv(T) * SVector(gibbs - 0.5 * v1^2, v1, -1)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    V, v1, T = prim
    rho = inv(V)
    rho_v1 = rho * v1
    e = energy_internal(V, T, eos)
    rho_e_total = rho * e + 0.5 * rho_v1 * v1
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
    S = -rho * specific_entropy(V, T, eos)
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
end # @muladd
