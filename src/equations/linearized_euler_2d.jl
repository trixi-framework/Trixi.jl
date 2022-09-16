# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    LinearizedEulerEquations2D(rho_mean, v1_mean, v2_mean, c)
        
Linearized euler equations in two space dimensions. The equations are given by
```math
\partial t
\begin{pmatrix}
    \rho' \\ v_1' \\ v_2' \\ p'
\end{pmatrix}
+
\partial x
\begin{pmatrix}
    \bar{\rho} v_1' + \bar{v_1} /rho' \\ \bar{v_1} v_1' + \frac{p'}{\bar{\rho}} \\ \bar{v_1} v_2' \\ \bar{v_1} p' + c^2 \bar{\rho} v_1'
\end{pmatrix}
+
\partial y
\begin{pmatrix}
    \bar{\rho} v_2' + \bar{v_2} /rho' \\ \bar{v_2} v_1' \\ \bar{v_2} v_2' + \frac{p'}{\bar{\rho}} \\ \bar{v_2} p' + c^2 \bar{\rho} v_2'
\end{pmatrix}
=
\begin{pmatrix}
    0 \\ 0 \\ 0 \\ 0
\end{pmatrix}
```
The bar ``\bar{(\cdot)}`` indicates uniform mean flow variables and c is the speed of sound. 
The unknowns are the acoustic velocities ``v_1', v_2'``, the pressure ``p'`` and the density ``\rho'``.
"""
struct LinearizedEulerEquations2D{RealT<:Real} <: AbstractLinearizedEulerEquations{2, 4}
    rho_mean::RealT
    v1_mean::RealT
    v2_mean::RealT
    c::RealT
  
    function LinearizedEulerEquations2D(rho_mean, v1_mean, v2_mean, c)
        if rho_mean < 0
          throw(ArgumentError("rho_mean must be non-negative"))
        elseif c < 0
          throw(ArgumentError("c must be non-negative"))
        end
        
        return new{typeof(rho_mean)}(rho_mean, v1_mean, v2_mean, c)
    end
end
    
function LinearizedEulerEquations2D(; rho_mean, v1_mean, v2_mean, c)
    return LinearizedEulerEquations2D(rho_mean, v1_mean, v2_mean, c)
end
    
    
varnames(::typeof(cons2cons), ::LinearizedEulerEquations2D) = ("rho_prime", "v1_prime", "v2_prime", "p_prime")
varnames(::typeof(cons2prim), ::LinearizedEulerEquations2D) = ("rho_prime", "v1_prime", "v2_prime", "p_prime")
    
"""
    initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations2D)
    
A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations2D)
    rho_prime = -cospi(2*t) * (sinpi(2*x[1]) + sinpi(2*x[2]))
    v1_prime = sinpi(2*t) * cospi(2*x[1])
    v2_prime = sinpi(2*t) * cospi(2*x[2])
    p_prime = rho_prime
      
    return SVector(rho_prime, v1_prime, v2_prime, p_prime)
end
    
    
"""
    boundary_condition_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                                equations::LinearizedEulerEquations2D)
    
Boundary conditions for a solid wall.
"""
function boundary_condition_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                                    equations::LinearizedEulerEquations2D)
    # Boundary state is equal to the inner state except for the velocity. For boundaries
    # in the -x/+x direction, we multiply the velocity in the x direction by -1.
    # Similarly, for boundaries in the -y/+y direction, we multiply the velocity in the
    # y direction by -1
    if direction in (1, 2) # x direction
        u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4])
    else # y direction
        u_boundary = SVector(u_inner[1], u_inner[2], -u_inner[3], u_inner[4])
    end
      
    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
      
    return flux
end
    
    
# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::LinearizedEulerEquations2D)
    (; rho_mean, v1_mean, v2_mean, c) = equations
    rho_prime, v1_prime, v2_prime, p_prime = u
    if orientation == 1
        f1 = v1_mean * rho_prime + rho_mean * v1_prime
        f2 = v1_mean * v1_prime + p_prime / rho_mean
        f3 = v1_mean * v2_prime
        f4 = v1_mean * p_prime + c^2 * rho_mean * v1_prime
    else
        f1 = v2_mean * rho_prime + rho_mean * v2_prime
        f2 = v2_mean * v1_prime
        f3 = v2_mean * v2_prime + p_prime / rho_mean
        f4 = v2_mean * p_prime + c^2 * rho_mean * v2_prime
    end
      
    return SVector(f1, f2, f3, f4)
end
    
    
@inline have_constant_speed(::LinearizedEulerEquations2D) = Val(true)
    
@inline function max_abs_speeds(equations::LinearizedEulerEquations2D)
    (; v1_mean, v2_mean, c) = equations
    return abs(v1_mean) + c, abs(v2_mean) + c
end
    
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::LinearizedEulerEquations2D)
    (; v1_mean, v2_mean, c) = equations
    if orientation == 1
        return abs(v1_mean) + c
    else # orientation == 2
        return abs(v2_mean) + c
    end
end
    
    
# Convert conservative variables to primitive
@inline cons2prim(u, equations::LinearizedEulerEquations2D) = cons2cons(u, equations)
@inline cons2entropy(u, ::LinearizedEulerEquations2D) = u
    
# Nondimensional variables
@inline function cons2nondim(u, equations::LinearizedEulerEquations2D)
    (; rho_mean, c) = equations
    rho_prime, v1_prime, v2_prime, p_prime = u
      
    rho_tilde = rho_prime / rho_mean
    v1_tilde = v1_prime / c
    v2_tilde = v2_prime / c
    p_tilde = p_prime / (rho_mean * c * c)
      
    return SVector(rho_tilde, v1_tilde, v2_tilde, p_tilde)
end
    
@inline function nondim2cons(u, equations::LinearizedEulerEquations2D)
    (; rho_mean, c) = equations
    rho_tilde, v1_tilde, v2_tilde, p_tilde = u
      
    rho_prime = rho_tilde * rho_mean
    v1_prime = v1_tilde * c
    v2_prime = v2_tilde * c
    p_prime = p_tilde * c * c * rho_mean
      
    return SVector(rho_prime, v1_prime, v2_prime, p_prime)
end
    
varnames(::typeof(cons2nondim), ::LinearizedEulerEquations2D) = ("rho_tilde", "v1_tilde", "v2_tilde", "p_tilde")
    
    
end # muladd
    