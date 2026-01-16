# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    NonIdealCompressibleEulerEquations1D(gamma)

The compressible Euler equations
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
\rho \\ \rho v_1 \\ \rho E
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
\rho v_1 \\ \rho v_1^2 + p \\ (\rho E + p) v_1
\end{pmatrix}
=
\begin{pmatrix}
0 \\ 0 \\ 0
\end{pmatrix}
```
for a gas with pressure specified by some equation of state in one space dimension.
Here, ``\rho`` is the density, ``v_1`` the velocity, ``E`` the specific total energy, 
and the pressure in terms of specific volume `V = inv(rho)` and temperature `T` is given 
by some user-specified equation of state (EOS)
```math
p = p(V, T),
```
Similarly, the internal energy is specified by `e = e(V, T)`. 

Because of this, the primitive variables are also defined to be `V, v1, T` (instead of 
`rho, v1, p` for `CompressibleEulerEquations1D`). The implementation also assumes 
mass basis unless otherwise specified.     
"""
struct NonIdealCompressibleEulerEquations1D{EoS_T <: AbstractEquationOfState} <:
       AbstractCompressibleEulerEquations{1, 3}
    equation_of_state::EoS_T
end

function varnames(::typeof(cons2cons), ::NonIdealCompressibleEulerEquations1D)
    return ("rho", "rho_v1", "rho_e")
end
varnames(::typeof(cons2prim), ::NonIdealCompressibleEulerEquations1D) = ("V", "v1", "T")

"""
    initial_condition_constant(x, t, equations::NonIdealCompressibleEulerEquations1D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t,
                                    equations::NonIdealCompressibleEulerEquations1D)
    RealT = eltype(x)
    rho = 1
    rho_v1 = convert(RealT, 0.1)
    rho_e = 10
    return SVector(rho, rho_v1, rho_e)
end

"""
    boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function, equations::NonIdealCompressibleEulerEquations1D)
Determine the boundary numerical surface flux for a slip wall condition.
Imposes a zero normal velocity at the wall.
Density and pressure are taken from the internal solution state and pressure.
Should be used together with [`TreeMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, orientation,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::NonIdealCompressibleEulerEquations1D)
    # compute the primitive variables
    rho_local, v_normal, p_local = cons2prim(u_inner, equations)

    if isodd(direction) # flip sign of normal to make it outward pointing
        v_normal *= -1
    end

    # For the slip wall we directly set the flux as the normal velocity is zero
    return SVector(0, p_local, 0)
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer,
                      equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state

    rho, rho_v1, rho_E = u
    V, v1, T = cons2prim(u, equations)
    p = pressure(V, T, eos)

    # Ignore orientation since it is always "1" in 1D
    f1 = rho_v1
    f2 = rho_v1 * v1 + p
    f3 = (rho_E + p) * v1
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
    rho, rho_v1, rho_E = u

    V = inv(rho)
    v1 = rho_v1 * V
    e = (rho_E - 0.5 * rho_v1 * v1) * V
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
    e = internal_energy(V, T, eos)
    rho_E = rho * e + 0.5 * rho_v1 * v1
    return SVector(rho, rho_v1, rho_E)
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
    V, v1, T = cons2prim(u, equations)
    rho = u[1]
    S = -rho * specific_entropy(V, T, eos)
    return S
end

"""
    entropy(cons, equations::AbstractCompressibleEulerEquations)

Default entropy is the mathematical entropy
[`entropy_math(cons, equations::AbstractCompressibleEulerEquations)`](@ref).
"""
@inline function entropy(cons, equations::NonIdealCompressibleEulerEquations1D)
    return entropy_math(cons, equations)
end
end # @muladd
