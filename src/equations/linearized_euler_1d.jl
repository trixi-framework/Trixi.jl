# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LinearizedEulerEquations1D(v_mean_global, c_mean_global, rho_mean_global)

Linearized Euler equations in one space dimension. The equations are given by
```math
\partial_t
\begin{pmatrix}
    \rho' \\ v_1' \\ p'
\end{pmatrix}
+
\partial_x
\begin{pmatrix}
    \bar{\rho} v_1' + \bar{v_1} \rho ' \\ \bar{v_1} v_1' + \frac{p'}{\bar{\rho}} \\ \bar{v_1} p' + c^2 \bar{\rho} v_1'
\end{pmatrix}
=
\begin{pmatrix}
    0 \\ 0 \\ 0
\end{pmatrix}
```
The bar ``\bar{(\cdot)}`` indicates uniform mean flow variables and ``c`` is the speed of sound.
The unknowns are the perturbation quantities of the acoustic velocity ``v_1'``, the pressure ``p'`` 
and the density ``\rho'``.
"""
struct LinearizedEulerEquations1D{RealT <: Real} <:
       AbstractLinearizedEulerEquations{1, 3}
    v_mean_global::RealT
    c_mean_global::RealT
    rho_mean_global::RealT
end

function LinearizedEulerEquations1D(v_mean_global::Real,
                                    c_mean_global::Real, rho_mean_global::Real)
    if rho_mean_global < 0
        throw(ArgumentError("rho_mean_global must be non-negative"))
    elseif c_mean_global < 0
        throw(ArgumentError("c_mean_global must be non-negative"))
    end

    return LinearizedEulerEquations1D(v_mean_global, c_mean_global,
                                      rho_mean_global)
end

# Constructor with keywords
function LinearizedEulerEquations1D(; v_mean_global::Real,
                                    c_mean_global::Real, rho_mean_global::Real)
    return LinearizedEulerEquations1D(v_mean_global, c_mean_global,
                                      rho_mean_global)
end

function varnames(::typeof(cons2cons), ::LinearizedEulerEquations1D)
    ("rho_prime", "v1_prime", "p_prime")
end
function varnames(::typeof(cons2prim), ::LinearizedEulerEquations1D)
    ("rho_prime", "v1_prime", "p_prime")
end

"""
    initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations1D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations1D)
    rho_prime = -cospi(2 * t) * sinpi(2 * x[1])
    v1_prime = sinpi(2 * t) * cospi(2 * x[1])
    p_prime = rho_prime

    return SVector(rho_prime, v1_prime, p_prime)
end

"""
    boundary_condition_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                                equations::LinearizedEulerEquations1D)

Boundary conditions for a solid wall.
"""
function boundary_condition_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function,
                                 equations::LinearizedEulerEquations1D)
    # Boundary state is equal to the inner state except for the velocity. For boundaries
    # in the -x/+x direction, we multiply the velocity (in the x direction by) -1.
    u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3])

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::LinearizedEulerEquations1D)
    @unpack v_mean_global, c_mean_global, rho_mean_global = equations
    rho_prime, v1_prime, p_prime = u
    f1 = v_mean_global * rho_prime + rho_mean_global * v1_prime
    f2 = v_mean_global * v1_prime + p_prime / rho_mean_global
    f3 = v_mean_global * p_prime + c_mean_global^2 * rho_mean_global * v1_prime

    return SVector(f1, f2, f3)
end

@inline have_constant_speed(::LinearizedEulerEquations1D) = True()

@inline function max_abs_speeds(equations::LinearizedEulerEquations1D)
    @unpack v_mean_global, c_mean_global = equations
    return abs(v_mean_global) + c_mean_global
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedEulerEquations1D)
    @unpack v_mean_global, c_mean_global = equations
    return abs(v_mean_global) + c_mean_global
end

# Calculate estimate for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedEulerEquations1D)
    min_max_speed_davis(u_ll, u_rr, orientation, equations)
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedEulerEquations1D)
    @unpack v_mean_global, c_mean_global = equations

    λ_min = v_mean_global - c_mean_global
    λ_max = v_mean_global + c_mean_global

    return λ_min, λ_max
end

# Convert conservative variables to primitive
@inline cons2prim(u, equations::LinearizedEulerEquations1D) = u
@inline cons2entropy(u, ::LinearizedEulerEquations1D) = u
end # muladd
