# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    NonIdealCompressibleEulerEquations2D(equation_of_state)

The compressible Euler equations
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
    \rho \\ \rho v_1 \\ \rho v_2 \\ \rho e_{total}
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
 \rho v_1 \\ \rho v_1^2 + p \\ \rho v_1 v_2 \\ (\rho e_{total} + p) v_1
\end{pmatrix}
+
\frac{\partial}{\partial y}
\begin{pmatrix}
\rho v_2 \\ \rho v_1 v_2 \\ \rho v_2^2 + p \\ (\rho e_{total} + p) v_2
\end{pmatrix}
=
\begin{pmatrix}
    0 \\ 0 \\ 0
\end{pmatrix}
```
for a gas with pressure ``p`` specified by some equation of state in one space dimension.

Here, ``\rho`` is the density, ``v_1`` the x-velocity, ``v_2`` is the y-velocity, ``e_{total}`` 
the specific total energy, and the pressure ``p`` is given in terms of specific volume ``V = 1/\rho`` 
and temperature ``T`` by some user-specified equation of state (EOS) (see [`pressure(V, T, eos::IdealGas)`](@ref), 
[`pressure(V, T, eos::VanDerWaals)`](@ref)) as
```math
p = p(V, T)
```

Similarly, the internal energy is specified by `e_{\text{internal}} = energy_internal(V, T, eos)`, see
[`energy_internal(V, T, eos::IdealGas)`](@ref), [`energy_internal(V, T, eos::VanDerWaals)`](@ref).

Because of this, the primitive variables are also defined to be `V, v1, v2, T` (instead of 
`rho, v1, v2, p` for `CompressibleEulerEquations2D`). The implementation also assumes 
mass basis unless otherwise specified.     
"""
struct NonIdealCompressibleEulerEquations2D{EoS <: AbstractEquationOfState} <:
       AbstractNonIdealCompressibleEulerEquations{2, 4}
    equation_of_state::EoS
end

function varnames(::typeof(cons2cons), ::NonIdealCompressibleEulerEquations2D)
    return ("rho", "rho_v1", "rho_v2", "rho_e_total")
end

# for plotting with PlotData1D(sol, solution_variables=density_velocity_pressure)
@inline function cons2prim(u, equations::NonIdealCompressibleEulerEquations2D)
    eos = equations.equation_of_state
    rho = u[1]
    V, v1, v2, T = cons2thermo(u, equations)
    return SVector(rho, v1, v2, pressure(V, T, eos))
end
varnames(::typeof(cons2prim), ::NonIdealCompressibleEulerEquations2D) = ("rho",
                                                                         "v1",
                                                                         "v2",
                                                                         "p")

# Calculate flux for a single point
@inline function flux(u, orientation::Integer,
                      equations::NonIdealCompressibleEulerEquations2D)
    eos = equations.equation_of_state

    _, rho_v1, rho_v2, rho_e_total = u
    V, v1, v2, T = cons2thermo(u, equations)
    p = pressure(V, T, eos)

    if orientation == 1
        f1 = rho_v1
        f2 = rho_v1 * v1 + p
        f3 = rho_v1 * v2
        f4 = (rho_e_total + p) * v1
    else
        f1 = rho_v2
        f2 = rho_v2 * v1
        f3 = rho_v2 * v2 + p
        f4 = (rho_e_total + p) * v2
    end
    return SVector(f1, f2, f3, f4)
end

# Calculate 2D flux for a single point
@inline function flux(u, normal_direction::AbstractVector,
                      equations::NonIdealCompressibleEulerEquations2D)
    eos = equations.equation_of_state

    rho = first(u)
    rho_e_total = last(u)
    V, v1, v2, T = cons2thermo(u, equations)
    p = pressure(V, T, eos)

    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
    rho_v_normal = rho * v_normal

    f1 = rho_v_normal
    f2 = rho_v_normal * v1 + p * normal_direction[1]
    f3 = rho_v_normal * v2 + p * normal_direction[2]
    f4 = (rho_e_total + p) * v_normal
    return SVector(f1, f2, f3, f4)
end

# the default amplitude and frequency k are chosen to be consistent with 
# initial_condition_density_wave for CompressibleEulerEquations1D
function initial_condition_density_wave(x, t,
                                        equations::NonIdealCompressibleEulerEquations2D;
                                        amplitude = 0.98, k = 2)
    RealT = eltype(x)

    eos = equations.equation_of_state

    v1 = convert(RealT, 0.1)
    v2 = convert(RealT, 0.2)
    rho = 1 + convert(RealT, amplitude) * sinpi(k * (x[1] + x[2] - t * (v1 + v2)))
    p = 20

    V = inv(rho)

    # invert for temperature given p, V
    T = 1
    tol = 100 * eps(RealT)
    dp = pressure(V, T, eos) - p
    iter = 1
    while abs(dp) / abs(p) > tol && iter < 100
        dp = pressure(V, T, eos) - p
        dpdT_V = ForwardDiff.derivative(T -> pressure(V, T, eos), T)
        T = max(tol, T - dp / dpdT_V)
        iter += 1
    end
    if iter == 100
        println("Warning: solver for temperature(V, p) did not converge")
    end

    return prim2cons(SVector(V, v1, v2, T), equations)
end

"""
    boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function, equations::NonIdealCompressibleEulerEquations2D)

Should be used together with [`TreeMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, orientation,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::NonIdealCompressibleEulerEquations2D)
    # get the appropriate normal vector from the orientation
    RealT = eltype(u_inner)
    if orientation == 1
        normal_direction = SVector(one(RealT), zero(RealT))
    else # orientation == 2
        normal_direction = SVector(zero(RealT), one(RealT))
    end

    # compute and return the flux using `boundary_condition_slip_wall` routine below
    return boundary_condition_slip_wall(u_inner, normal_direction,
                                        x, t, surface_flux_function, equations)
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                 equations::NonIdealCompressibleEulerEquations2D)

Determine the boundary numerical surface flux for a slip wall condition.
Imposes a zero normal velocity at the wall.
Density is taken from the internal solution state, 

Should be used together with [`UnstructuredMesh2D`](@ref), [`P4estMesh`](@ref), or [`T8codeMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              x, t, surface_flux_function,
                                              equations::NonIdealCompressibleEulerEquations2D)
    p = pressure(u_inner, equations)

    # For the slip wall we directly set the flux as the normal velocity is zero
    return SVector(0,
                   p * normal_direction[1],
                   p * normal_direction[2],
                   0)
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, direction, x, t,
                                 surface_flux_function, equations::CompressibleEulerEquations2D)

Should be used together with [`StructuredMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::NonIdealCompressibleEulerEquations2D)
    # flip sign of normal to make it outward pointing, then flip the sign of the normal flux back
    # to be inward pointing on the -x and -y sides due to the orientation convention used by StructuredMesh
    if isodd(direction)
        boundary_flux = -boundary_condition_slip_wall(u_inner, -normal_direction,
                                                      x, t, surface_flux_function,
                                                      equations)
    else
        boundary_flux = boundary_condition_slip_wall(u_inner, normal_direction,
                                                     x, t, surface_flux_function,
                                                     equations)
    end

    return boundary_flux
end

"""
    flux_terashima_etal(u_ll, u_rr, orientation::Int,
                        equations::NonIdealCompressibleEulerEquations1D)

Approximately pressure equilibrium conserving (APEC) flux from 
"Approximately pressure-equilibrium-preserving scheme for fully conservative 
simulations of compressible multi-species and real-fluid interfacial flows" 
by Terashima, Ly, Ihme (2025). https://doi.org/10.1016/j.jcp.2024.11370 1

"""
function flux_terashima_etal(u_ll, u_rr, orientation::Int,
                             equations::NonIdealCompressibleEulerEquations2D)
    eos = equations.equation_of_state
    V_ll, v1_ll, v2_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, v2_rr, T_rr = cons2thermo(u_rr, equations)

    rho_ll = u_ll[1]
    rho_rr = u_rr[1]
    rho_e_ll = energy_internal(u_ll, equations)
    rho_e_rr = energy_internal(u_rr, equations)
    p_ll = pressure(V_ll, T_ll, eos)
    p_rr = pressure(V_rr, T_rr, eos)

    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    rho_e_avg = 0.5f0 * (rho_e_ll + rho_e_rr)
    p_v1_avg = 0.5f0 * (p_ll * v1_rr + p_rr * v1_ll)
    p_v2_avg = 0.5f0 * (p_ll * v2_rr + p_rr * v2_ll)

    # chain rule from Terashima    
    drho_e_drho_p_ll = drho_e_drho_at_const_p(V_ll, T_ll, eos)
    drho_e_drho_p_rr = drho_e_drho_at_const_p(V_rr, T_rr, eos)
    rho_e_avg_corrected = (rho_e_avg -
                           0.25f0 * (drho_e_drho_p_rr - drho_e_drho_p_ll) *
                           (rho_rr - rho_ll))
    ke_avg = 0.5f0 * ((v1_ll * v1_rr) + (v2_ll * v2_rr))

    if orientation == 1
        f_rho = rho_avg * v1_avg
        f_rho_v1 = f_rho * v1_avg + p_avg
        f_rho_v2 = f_rho * v2_avg
        f_rho_E = (rho_e_avg_corrected + rho_avg * ke_avg) * v1_avg + p_v1_avg
    else # if orientation == 2
        f_rho = rho_avg * v2_avg
        f_rho_v1 = f_rho * v1_avg
        f_rho_v2 = f_rho * v2_avg + p_avg
        f_rho_E = (rho_e_avg_corrected + rho_avg * ke_avg) * v2_avg + p_v2_avg
    end

    return SVector(f_rho, f_rho_v1, f_rho_v2, f_rho_E)
end

function flux_terashima_etal(u_ll, u_rr, normal_direction::AbstractVector,
                             equations::NonIdealCompressibleEulerEquations2D)
    eos = equations.equation_of_state
    V_ll, v1_ll, v2_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, v2_rr, T_rr = cons2thermo(u_rr, equations)

    rho_ll = u_ll[1]
    rho_rr = u_rr[1]
    rho_e_ll = energy_internal(u_ll, equations)
    rho_e_rr = energy_internal(u_rr, equations)
    p_ll = pressure(V_ll, T_ll, eos)
    p_rr = pressure(V_rr, T_rr, eos)

    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
    v_dot_n_avg = 0.5f0 * (v_dot_n_ll + v_dot_n_rr)

    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    rho_e_avg = 0.5f0 * (rho_e_ll + rho_e_rr)
    p_v_dot_n_avg = 0.5f0 * (p_ll * v_dot_n_rr + p_rr * v_dot_n_ll)

    # chain rule from Terashima    
    drho_e_drho_p_ll = drho_e_drho_at_const_p(V_ll, T_ll, eos)
    drho_e_drho_p_rr = drho_e_drho_at_const_p(V_rr, T_rr, eos)
    rho_e_avg_corrected = (rho_e_avg -
                           0.25f0 * (drho_e_drho_p_rr - drho_e_drho_p_ll) *
                           (rho_rr - rho_ll))

    ke_avg = 0.5f0 * ((v1_ll * v1_rr) + (v2_ll * v2_rr))

    f_rho = rho_avg * v_dot_n_avg
    f_rho_v1 = f_rho * v1_avg + p_avg * normal_direction[1]
    f_rho_v2 = f_rho * v2_avg + p_avg * normal_direction[2]
    f_rho_E = (rho_e_avg_corrected + rho_avg * ke_avg) * v_dot_n_avg + p_v_dot_n_avg
    return SVector(f_rho, f_rho_v1, f_rho_v2, f_rho_E)
end

"""
    flux_central_terashima_etal(u_ll, u_rr, orientation::Int,
                                equations::NonIdealCompressibleEulerEquations1D)

A version of the central flux which uses the approximately pressure equilibrium conserving 
(APEC) internal energy correction of 
"Approximately pressure-equilibrium-preserving scheme for fully conservative 
simulations of compressible multi-species and real-fluid interfacial flows" 
by Terashima, Ly, Ihme (2025). https://doi.org/10.1016/j.jcp.2024.11370 
"""
function flux_central_terashima_etal(u_ll, u_rr, orientation::Int,
                                     equations::NonIdealCompressibleEulerEquations2D)
    eos = equations.equation_of_state
    V_ll, v1_ll, v2_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, v2_rr, T_rr = cons2thermo(u_rr, equations)

    rho_ll, rho_v1_ll, rho_v2_ll, rho_e_total_ll = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, rho_e_total_rr = u_rr
    rho_e_ll = energy_internal(u_ll, equations)
    rho_e_rr = energy_internal(u_rr, equations)
    p_ll = pressure(V_ll, T_ll, eos)
    p_rr = pressure(V_rr, T_rr, eos)

    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    rho_e_avg = 0.5f0 * (rho_e_ll + rho_e_rr)

    # chain rule from Terashima    
    drho_e_drho_p_ll = drho_e_drho_at_const_p(V_ll, T_ll, eos)
    drho_e_drho_p_rr = drho_e_drho_at_const_p(V_rr, T_rr, eos)
    rho_e_avg_corrected = (rho_e_avg -
                           0.25f0 * (drho_e_drho_p_rr - drho_e_drho_p_ll) *
                           (rho_rr - rho_ll))

    # calculate internal energy (with APEC correction) and kinetic energy 
    # contributions separately in energy equation
    ke_ll = 0.5f0 * (v1_ll^2 + v2_ll^2)
    ke_rr = 0.5f0 * (v1_rr^2 + v2_rr^2)

    if orientation == 1
        f_rho = 0.5f0 * (rho_v1_ll + rho_v1_rr)
        f_rho_v1 = 0.5f0 * (rho_v1_ll * v1_ll + rho_v1_rr * v1_rr) + p_avg
        f_rho_v2 = 0.5f0 * (rho_v1_ll * v2_ll + rho_v1_rr * v2_rr)
        f_rho_E = rho_e_avg_corrected * v1_avg +
                  0.5f0 * (rho_v1_ll * ke_ll + rho_v1_rr * ke_rr) +
                  0.5f0 * (p_ll * v1_ll + p_rr * v1_rr)
    else # if orientation == 2
        f_rho = 0.5f0 * (rho_v2_ll + rho_v2_rr)
        f_rho_v1 = 0.5f0 * (rho_v1_ll * v2_ll + rho_v1_rr * v2_rr)
        f_rho_v2 = 0.5f0 * (rho_v2_ll * v2_ll + rho_v2_rr * v2_rr) + p_avg
        f_rho_E = rho_e_avg_corrected * v2_avg +
                  0.5f0 * (rho_v2_ll * ke_ll + rho_v2_rr * ke_rr) +
                  0.5f0 * (p_ll * v2_ll + p_rr * v2_rr)
    end

    return SVector(f_rho, f_rho_v1, f_rho_v2, f_rho_E)
end

function flux_central_terashima_etal(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::NonIdealCompressibleEulerEquations2D)
    eos = equations.equation_of_state
    V_ll, v1_ll, v2_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, v2_rr, T_rr = cons2thermo(u_rr, equations)

    rho_ll, rho_v1_ll, rho_v2_ll, rho_e_total_ll = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, rho_e_total_rr = u_rr
    rho_e_ll = energy_internal(u_ll, equations)
    rho_e_rr = energy_internal(u_rr, equations)
    p_ll = pressure(V_ll, T_ll, eos)
    p_rr = pressure(V_rr, T_rr, eos)

    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
    v_dot_n_avg = 0.5f0 * (v_dot_n_ll + v_dot_n_rr)

    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    rho_e_avg = 0.5f0 * (rho_e_ll + rho_e_rr)
    p_v_dot_n_avg = 0.5f0 * (p_ll * v_dot_n_rr + p_rr * v_dot_n_ll)

    # chain rule from Terashima    
    drho_e_drho_p_ll = drho_e_drho_at_const_p(V_ll, T_ll, eos)
    drho_e_drho_p_rr = drho_e_drho_at_const_p(V_rr, T_rr, eos)
    rho_e_avg_corrected = (rho_e_avg -
                           0.25f0 * (drho_e_drho_p_rr - drho_e_drho_p_ll) *
                           (rho_rr - rho_ll))

    # calculate internal energy (with APEC correction) and kinetic energy 
    # contributions separately in energy equation
    ke_ll = 0.5f0 * (v1_ll^2 + v2_ll^2)
    ke_rr = 0.5f0 * (v1_rr^2 + v2_rr^2)

    rho_v_dot_n_ll = rho_ll * v_dot_n_ll
    rho_v_dot_n_rr = rho_rr * v_dot_n_rr
    f_rho = 0.5f0 * (rho_v_dot_n_ll + rho_v_dot_n_rr)
    f_rho_v1 = 0.5f0 * (rho_v_dot_n_ll * v1_ll + rho_v_dot_n_rr * v1_rr) +
               p_avg * normal_direction[1]
    f_rho_v2 = 0.5f0 * (rho_v_dot_n_ll * v2_ll + rho_v_dot_n_rr * v2_rr) +
               p_avg * normal_direction[2]
    f_rho_E = rho_e_avg_corrected * v_dot_n_avg +
              0.5f0 * (rho_v_dot_n_ll * ke_ll + rho_v_dot_n_rr * ke_rr) +
              0.5f0 * (p_ll * v_dot_n_ll + p_rr * v_dot_n_rr)
    return SVector(f_rho, f_rho_v1, f_rho_v2, f_rho_E)
end

# Calculate estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::NonIdealCompressibleEulerEquations2D)
    V_ll, v1_ll, v2_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, v2_rr, T_rr = cons2thermo(u_rr, equations)

    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    if orientation == 1 # x-direction
        λ_min = v1_ll - c_ll
        λ_max = v1_rr + c_rr
    else # y-direction
        λ_min = v2_ll - c_ll
        λ_max = v2_rr + c_rr
    end

    return λ_min, λ_max
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::NonIdealCompressibleEulerEquations2D)
    V_ll, v1_ll, v2_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, v2_rr, T_rr = cons2thermo(u_rr, equations)

    v_normal_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_normal_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    norm_ = norm(normal_direction)
    # The v_normals are already scaled by the norm
    λ_min = v_normal_ll - c_ll * norm_
    λ_max = v_normal_rr + c_rr * norm_

    return λ_min, λ_max
end

# Less "cautious", i.e., less overestimating `λ_max` compared to `max_abs_speed_naive`
@inline function max_abs_speed(u_ll, u_rr, orientation::Integer,
                               equations::NonIdealCompressibleEulerEquations2D)
    V_ll, v1_ll, v2_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, v2_rr, T_rr = cons2thermo(u_rr, equations)

    # Get the velocity value in the appropriate direction
    if orientation == 1
        v_ll = v1_ll
        v_rr = v1_rr
    else # orientation == 2
        v_ll = v2_ll
        v_rr = v2_rr
    end
    v_mag_ll = abs(v_ll)
    v_mag_rr = abs(v_rr)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    return max(v_mag_ll + c_ll, v_mag_rr + c_rr)
end

@inline function max_abs_speed(u_ll, u_rr, normal_direction::AbstractVector,
                               equations::NonIdealCompressibleEulerEquations2D)
    V_ll, v1_ll, v2_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, v2_rr, T_rr = cons2thermo(u_rr, equations)

    # Calculate normal velocities and sound speeds
    # left
    v_ll = (v1_ll * normal_direction[1]
            +
            v2_ll * normal_direction[2])

    # right
    v_rr = (v1_rr * normal_direction[1]
            +
            v2_rr * normal_direction[2])

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    norm_ = norm(normal_direction)
    return max(abs(v_ll) + c_ll * norm_,
               abs(v_rr) + c_rr * norm_)
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::NonIdealCompressibleEulerEquations2D)
    V_ll, v1_ll, v2_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, v2_rr, T_rr = cons2thermo(u_rr, equations)

    # Get the velocity value in the appropriate direction
    if orientation == 1
        v_ll = v1_ll
        v_rr = v1_rr
    else # orientation == 2
        v_ll = v2_ll
        v_rr = v2_rr
    end

    v_mag_ll = abs(v_ll)
    v_mag_rr = abs(v_rr)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    return max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::NonIdealCompressibleEulerEquations2D)
    V_ll, v1_ll, v2_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, v2_rr, T_rr = cons2thermo(u_rr, equations)

    # Get the velocity value in the appropriate direction
    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    v_mag_ll = abs(v_dot_n_ll)
    v_mag_rr = abs(v_dot_n_rr)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    return max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::NonIdealCompressibleEulerEquations2D)
    V_ll, v1_ll, v2_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, v2_rr, T_rr = cons2thermo(u_rr, equations)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    if orientation == 1 # x-direction
        λ_min = min(v1_ll - c_ll, v1_rr - c_rr)
        λ_max = max(v1_ll + c_ll, v1_rr + c_rr)
    else # y-direction
        λ_min = min(v2_ll - c_ll, v2_rr - c_rr)
        λ_max = max(v2_ll + c_ll, v2_rr + c_rr)
    end

    return λ_min, λ_max
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::NonIdealCompressibleEulerEquations2D)
    V_ll, v1_ll, v2_ll, T_ll = cons2thermo(u_ll, equations)
    V_rr, v1_rr, v2_rr, T_rr = cons2thermo(u_rr, equations)

    norm_ = norm(normal_direction)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos) * norm_
    c_rr = speed_of_sound(V_rr, T_rr, eos) * norm_

    v_normal_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_normal_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    # The v_normals are already scaled by the norm
    λ_min = min(v_normal_ll - c_ll, v_normal_rr - c_rr)
    λ_max = max(v_normal_ll + c_ll, v_normal_rr + c_rr)

    return λ_min, λ_max
end

@inline function max_abs_speeds(u, equations::NonIdealCompressibleEulerEquations2D)
    V, v1, v2, T = cons2thermo(u, equations)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c = speed_of_sound(V, T, eos)

    return (abs(v1) + c, abs(v2) + c)
end

"""
    function cons2thermo(u, equations::NonIdealCompressibleEulerEquations2D)
        
Convert conservative variables to specific volume, velocity, and temperature 
variables `V, v1, v2, T`. These are referred to as "thermodynamic" variables since
equation of state routines are assumed to be evaluated in terms of `V` and `T`. 
"""
@inline function cons2thermo(u, equations::NonIdealCompressibleEulerEquations2D)
    eos = equations.equation_of_state
    rho, rho_v1, rho_v2, rho_e_total = u

    V = inv(rho)
    v1 = rho_v1 * V
    v2 = rho_v2 * V
    e_internal = energy_internal(u, equations) * V
    T = temperature(V, e_internal, eos)

    return SVector(V, v1, v2, T)
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::NonIdealCompressibleEulerEquations2D)
    V, v1, v2, T = cons2thermo(u, equations)
    eos = equations.equation_of_state
    gibbs = gibbs_free_energy(V, T, eos)
    return inv(T) * SVector(gibbs - 0.5f0 * (v1^2 + v2^2), v1, v2, -1)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::NonIdealCompressibleEulerEquations2D)
    eos = equations.equation_of_state
    V, v1, v2, T = prim
    rho = inv(V)
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    e_internal = energy_internal_specific(V, T, eos)
    rho_e_total = rho * e_internal + 0.5f0 * (rho_v1 * v1 + rho_v2 * v2)
    return SVector(rho, rho_v1, rho_v2, rho_e_total)
end

@inline function velocity(u, orientation::Int,
                          equations::NonIdealCompressibleEulerEquations2D)
    if orientation == 1
        v1 = u[2] / rho
        return v1
    else # if orientation == 2
        v2 = u[3] / rho
        return v2
    end
end

@inline function velocity(u, equations::NonIdealCompressibleEulerEquations2D)
    rho = u[1]
    v1 = u[2] / rho
    v2 = u[3] / rho
    return SVector(v1, v2)
end

@inline function energy_internal(u,
                                 equations::NonIdealCompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e_total = u
    rho_e = rho_e_total - 0.5f0 * (rho_v1^2 + rho_v2^2) / rho
    return rho_e
end
end # @muladd
