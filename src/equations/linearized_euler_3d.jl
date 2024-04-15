# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LinearizedEulerEquations3D(v_mean_global, c_mean_global, rho_mean_global)

Linearized Euler equations in three space dimensions. The equations are given by
```math
\partial_t
\begin{pmatrix}
    \rho' \\ v_1' \\ v_2' \\ v_3' \\ p'
\end{pmatrix}
+
\partial_x
\begin{pmatrix}
    \bar{\rho} v_1' + \bar{v_1} \rho ' \\ 
    \bar{v_1} v_1' + \frac{p'}{\bar{\rho}} \\ 
    \bar{v_1} v_2' \\ 
    \bar{v_1} v_3' \\ 
    \bar{v_1} p' + c^2 \bar{\rho} v_1'
\end{pmatrix}
+
\partial_y
\begin{pmatrix}
    \bar{\rho} v_2' + \bar{v_2} \rho ' \\ 
    \bar{v_2} v_1' \\ 
    \bar{v_2} v_2' + \frac{p'}{\bar{\rho}} \\ 
    \bar{v_2} v_3' \\ 
    \bar{v_2} p' + c^2 \bar{\rho} v_2'
\end{pmatrix}
+ 
\partial_z
\begin{pmatrix}
    \bar{\rho} v_3' + \bar{v_3} \rho ' \\ 
    \bar{v_3} v_1' \\ 
    \bar{v_3} v_2' \\ 
    \bar{v_3} v_3' + \frac{p'}{\bar{\rho}} \\ 
    \bar{v_3} p' + c^2 \bar{\rho} v_3'
\end{pmatrix}
=
\begin{pmatrix}
    0 \\ 0 \\ 0 \\ 0 \\ 0
\end{pmatrix}
```
The bar ``\bar{(\cdot)}`` indicates uniform mean flow variables and ``c`` is the speed of sound.
The unknowns are the acoustic velocities ``v' = (v_1', v_2, v_3')``, the pressure ``p'`` and the density ``\rho'``.
"""
struct LinearizedEulerEquations3D{RealT <: Real} <:
       AbstractLinearizedEulerEquations{3, 5}
    v_mean_global::SVector{3, RealT}
    c_mean_global::RealT
    rho_mean_global::RealT
end

function LinearizedEulerEquations3D(v_mean_global::NTuple{3, <:Real},
                                    c_mean_global::Real, rho_mean_global::Real)
    if rho_mean_global < 0
        throw(ArgumentError("rho_mean_global must be non-negative"))
    elseif c_mean_global < 0
        throw(ArgumentError("c_mean_global must be non-negative"))
    end

    return LinearizedEulerEquations3D(SVector(v_mean_global), c_mean_global,
                                      rho_mean_global)
end

function LinearizedEulerEquations3D(; v_mean_global::NTuple{3, <:Real},
                                    c_mean_global::Real, rho_mean_global::Real)
    return LinearizedEulerEquations3D(v_mean_global, c_mean_global,
                                      rho_mean_global)
end

function varnames(::typeof(cons2cons), ::LinearizedEulerEquations3D)
    ("rho_prime", "v1_prime", "v2_prime", "v3_prime", "p_prime")
end
function varnames(::typeof(cons2prim), ::LinearizedEulerEquations3D)
    ("rho_prime", "v1_prime", "v2_prime", "v3_prime", "p_prime")
end

"""
    initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations3D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations3D)
    rho_prime = -cospi(2 * t) * (sinpi(2 * x[1]) + sinpi(2 * x[2]) + sinpi(2 * x[3]))
    v1_prime = sinpi(2 * t) * cospi(2 * x[1])
    v2_prime = sinpi(2 * t) * cospi(2 * x[2])
    v3_prime = sinpi(2 * t) * cospi(2 * x[3])
    p_prime = rho_prime

    return SVector(rho_prime, v1_prime, v2_prime, v3_prime, p_prime)
end

"""
    boundary_condition_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                                equations::LinearizedEulerEquations3D)

Boundary conditions for a solid wall.
"""
function boundary_condition_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function,
                                 equations::LinearizedEulerEquations3D)
    # Boundary state is equal to the inner state except for the velocity. For boundaries
    # in the -x/+x direction, we multiply the velocity in the x direction by -1.
    # Similarly, for boundaries in the -y/+y or -z/+z direction, we multiply the
    # velocity in the y or z direction by -1
    if direction in (1, 2) # x direction
        u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4],
                             u_inner[5])
    elseif direction in (3, 4) # y direction
        u_boundary = SVector(u_inner[1], u_inner[2], -u_inner[3], u_inner[4],
                             u_inner[5])
    else # z direction = (5, 6)
        u_boundary = SVector(u_inner[1], u_inner[2], u_inner[3], -u_inner[4],
                             u_inner[5])
    end

    # Calculate boundary flux depending on the orientation of the boundary
    # Odd directions are in negative coordinate direction, 
    # even directions are in positive coordinate direction.
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end

# Calculate 3D flux for a single point
@inline function flux(u, orientation::Integer, equations::LinearizedEulerEquations3D)
    @unpack v_mean_global, c_mean_global, rho_mean_global = equations
    rho_prime, v1_prime, v2_prime, v3_prime, p_prime = u
    if orientation == 1
        f1 = v_mean_global[1] * rho_prime + rho_mean_global * v1_prime
        f2 = v_mean_global[1] * v1_prime + p_prime / rho_mean_global
        f3 = v_mean_global[1] * v2_prime
        f4 = v_mean_global[1] * v3_prime
        f5 = v_mean_global[1] * p_prime + c_mean_global^2 * rho_mean_global * v1_prime
    elseif orientation == 2
        f1 = v_mean_global[2] * rho_prime + rho_mean_global * v2_prime
        f2 = v_mean_global[2] * v1_prime
        f3 = v_mean_global[2] * v2_prime + p_prime / rho_mean_global
        f4 = v_mean_global[2] * v3_prime
        f5 = v_mean_global[2] * p_prime + c_mean_global^2 * rho_mean_global * v2_prime
    else # orientation == 3
        f1 = v_mean_global[3] * rho_prime + rho_mean_global * v3_prime
        f2 = v_mean_global[3] * v1_prime
        f3 = v_mean_global[3] * v2_prime
        f4 = v_mean_global[3] * v3_prime + p_prime / rho_mean_global
        f5 = v_mean_global[3] * p_prime + c_mean_global^2 * rho_mean_global * v3_prime
    end

    return SVector(f1, f2, f3, f4, f5)
end

# Calculate 3D flux for a single point
@inline function flux(u, normal_direction::AbstractVector,
                      equations::LinearizedEulerEquations3D)
    @unpack v_mean_global, c_mean_global, rho_mean_global = equations
    rho_prime, v1_prime, v2_prime, v3_prime, p_prime = u

    v_mean_normal = v_mean_global[1] * normal_direction[1] +
                    v_mean_global[2] * normal_direction[2] +
                    v_mean_global[3] * normal_direction[3]
    v_prime_normal = v1_prime * normal_direction[1] + v2_prime * normal_direction[2] +
                     v3_prime * normal_direction[3]

    f1 = v_mean_normal * rho_prime + rho_mean_global * v_prime_normal
    f2 = v_mean_normal * v1_prime + normal_direction[1] * p_prime / rho_mean_global
    f3 = v_mean_normal * v2_prime + normal_direction[2] * p_prime / rho_mean_global
    f4 = v_mean_normal * v3_prime + normal_direction[3] * p_prime / rho_mean_global
    f5 = v_mean_normal * p_prime + c_mean_global^2 * rho_mean_global * v_prime_normal

    return SVector(f1, f2, f3, f4, f5)
end

@inline have_constant_speed(::LinearizedEulerEquations3D) = True()

@inline function max_abs_speeds(equations::LinearizedEulerEquations3D)
    @unpack v_mean_global, c_mean_global = equations
    return abs(v_mean_global[1]) + c_mean_global, abs(v_mean_global[2]) + c_mean_global,
           abs(v_mean_global[3]) + c_mean_global
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedEulerEquations3D)
    @unpack v_mean_global, c_mean_global = equations
    if orientation == 1
        return abs(v_mean_global[1]) + c_mean_global
    elseif orientation == 2
        return abs(v_mean_global[2]) + c_mean_global
    else # orientation == 3
        return abs(v_mean_global[3]) + c_mean_global
    end
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::LinearizedEulerEquations3D)
    @unpack v_mean_global, c_mean_global = equations
    v_mean_normal = normal_direction[1] * v_mean_global[1] +
                    normal_direction[2] * v_mean_global[2] +
                    normal_direction[3] * v_mean_global[3]
    return abs(v_mean_normal) + c_mean_global * norm(normal_direction)
end

# Calculate estimate for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedEulerEquations3D)
    min_max_speed_davis(u_ll, u_rr, orientation, equations)
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::LinearizedEulerEquations3D)
    min_max_speed_davis(u_ll, u_rr, normal_direction, equations)
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedEulerEquations3D)
    @unpack v_mean_global, c_mean_global = equations

    λ_min = v_mean_global[orientation] - c_mean_global
    λ_max = v_mean_global[orientation] + c_mean_global

    return λ_min, λ_max
end

@inline function min_max_speed_davis(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::LinearizedEulerEquations3D)
    @unpack v_mean_global, c_mean_global = equations

    norm_ = norm(normal_direction)

    v_normal = v_mean_global[1] * normal_direction[1] +
               v_mean_global[2] * normal_direction[2] +
               v_mean_global[3] * normal_direction[3]

    # The v_normals are already scaled by the norm
    λ_min = v_normal - c_mean_global * norm_
    λ_max = v_normal + c_mean_global * norm_

    return λ_min, λ_max
end

# Convert conservative variables to primitive
@inline cons2prim(u, equations::LinearizedEulerEquations3D) = u
@inline cons2entropy(u, ::LinearizedEulerEquations3D) = u
end # muladd
