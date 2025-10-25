# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LinearizedEulerEquations2D(v_mean_global, c_mean_global, rho_mean_global)

Linearized Euler equations in two space dimensions. The equations are given by
```math
\partial_t
\begin{pmatrix}
    \rho' \\ v_1' \\ v_2' \\ p'
\end{pmatrix}
+
\partial_x
\begin{pmatrix}
    \bar{\rho} v_1' + \bar{v_1} \rho ' \\ \bar{v_1} v_1' + \frac{p'}{\bar{\rho}} \\ \bar{v_1} v_2' \\ \bar{v_1} p' + c^2 \bar{\rho} v_1'
\end{pmatrix}
+
\partial_y
\begin{pmatrix}
    \bar{\rho} v_2' + \bar{v_2} \rho ' \\ \bar{v_2} v_1' \\ \bar{v_2} v_2' + \frac{p'}{\bar{\rho}} \\ \bar{v_2} p' + c^2 \bar{\rho} v_2'
\end{pmatrix}
=
\begin{pmatrix}
    0 \\ 0 \\ 0 \\ 0
\end{pmatrix}
```
The bar ``\bar{(\cdot)}`` indicates uniform mean flow variables and ``c`` is the speed of sound.
The unknowns are the acoustic velocities ``v' = (v_1', v_2')``, the pressure ``p'`` and the density ``\rho'``.
"""
struct LinearizedEulerEquations2D{RealT <: Real} <:
       AbstractLinearizedEulerEquations{2, 4}
    v_mean_global::SVector{2, RealT}
    c_mean_global::RealT
    rho_mean_global::RealT
end

function LinearizedEulerEquations2D(v_mean_global::NTuple{2, <:Real},
                                    c_mean_global::Real, rho_mean_global::Real)
    if rho_mean_global < 0
        throw(ArgumentError("rho_mean_global must be non-negative"))
    elseif c_mean_global < 0
        throw(ArgumentError("c_mean_global must be non-negative"))
    end

    return LinearizedEulerEquations2D(SVector(v_mean_global), c_mean_global,
                                      rho_mean_global)
end

function LinearizedEulerEquations2D(; v_mean_global::NTuple{2, <:Real},
                                    c_mean_global::Real, rho_mean_global::Real)
    return LinearizedEulerEquations2D(v_mean_global, c_mean_global,
                                      rho_mean_global)
end

function varnames(::typeof(cons2cons), ::LinearizedEulerEquations2D)
    ("rho_prime", "v1_prime", "v2_prime", "p_prime")
end
function varnames(::typeof(cons2prim), ::LinearizedEulerEquations2D)
    ("rho_prime", "v1_prime", "v2_prime", "p_prime")
end

"""
    initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations2D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations2D)
    rho_prime = -cospi(2 * t) * (sinpi(2 * x[1]) + sinpi(2 * x[2]))
    v1_prime = sinpi(2 * t) * cospi(2 * x[1])
    v2_prime = sinpi(2 * t) * cospi(2 * x[2])
    p_prime = rho_prime

    return SVector(rho_prime, v1_prime, v2_prime, p_prime)
end

"""
    boundary_condition_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                                equations::LinearizedEulerEquations2D)

Boundary conditions for a solid wall.
"""
function boundary_condition_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function,
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
    @unpack v_mean_global, c_mean_global, rho_mean_global = equations
    rho_prime, v1_prime, v2_prime, p_prime = u
    if orientation == 1
        f1 = v_mean_global[1] * rho_prime + rho_mean_global * v1_prime
        f2 = v_mean_global[1] * v1_prime + p_prime / rho_mean_global
        f3 = v_mean_global[1] * v2_prime
        f4 = v_mean_global[1] * p_prime + c_mean_global^2 * rho_mean_global * v1_prime
    else
        f1 = v_mean_global[2] * rho_prime + rho_mean_global * v2_prime
        f2 = v_mean_global[2] * v1_prime
        f3 = v_mean_global[2] * v2_prime + p_prime / rho_mean_global
        f4 = v_mean_global[2] * p_prime + c_mean_global^2 * rho_mean_global * v2_prime
    end

    return SVector(f1, f2, f3, f4)
end

# Calculate 1D flux for a single point
@inline function flux(u, normal_direction::AbstractVector,
                      equations::LinearizedEulerEquations2D)
    @unpack v_mean_global, c_mean_global, rho_mean_global = equations
    rho_prime, v1_prime, v2_prime, p_prime = u

    v_mean_normal = v_mean_global[1] * normal_direction[1] +
                    v_mean_global[2] * normal_direction[2]
    v_prime_normal = v1_prime * normal_direction[1] + v2_prime * normal_direction[2]

    f1 = v_mean_normal * rho_prime + rho_mean_global * v_prime_normal
    f2 = v_mean_normal * v1_prime + normal_direction[1] * p_prime / rho_mean_global
    f3 = v_mean_normal * v2_prime + normal_direction[2] * p_prime / rho_mean_global
    f4 = v_mean_normal * p_prime + c_mean_global^2 * rho_mean_global * v_prime_normal

    return SVector(f1, f2, f3, f4)
end

@inline have_constant_speed(::LinearizedEulerEquations2D) = True()

@inline function max_abs_speeds(equations::LinearizedEulerEquations2D)
    @unpack v_mean_global, c_mean_global = equations
    return abs(v_mean_global[1]) + c_mean_global, abs(v_mean_global[2]) + c_mean_global
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedEulerEquations2D)
    @unpack v_mean_global, c_mean_global = equations
    if orientation == 1
        return abs(v_mean_global[1]) + c_mean_global
    else # orientation == 2
        return abs(v_mean_global[2]) + c_mean_global
    end
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::LinearizedEulerEquations2D)
    @unpack v_mean_global, c_mean_global = equations
    v_mean_normal = normal_direction[1] * v_mean_global[1] +
                    normal_direction[2] * v_mean_global[2]
    return abs(v_mean_normal) + c_mean_global * norm(normal_direction)
end

@doc raw"""
    flux_godunov(u_ll, u_rr, orientation_or_normal_direction,
                 equations::LinearizedEulerEquations2D)

An upwind flux for the linearized Euler equations based on diagonalization of the physical
flux matrix. Given the physical flux ``Au``, ``A=T \Lambda T^{-1}`` with
``\Lambda`` being a diagonal matrix that holds the eigenvalues of ``A``, decompose
``\Lambda = \Lambda^+ + \Lambda^-`` where ``\Lambda^+`` and ``\Lambda^-`` are diagonal
matrices holding the positive and negative eigenvalues of ``A``, respectively. Then for
left and right states ``u_L, u_R``, the numerical flux calculated by this function is given
by ``A^+ u_L + A^- u_R`` where ``A^{\pm} = T \Lambda^{\pm} T^{-1}``.

The diagonalization of the flux matrix can be found in
- R. F. Warming, Richard M. Beam and B. J. Hyett (1975)
  Diagonalization and simultaneous symmetrization of the gas-dynamic matrices
  [DOI: 10.1090/S0025-5718-1975-0388967-5](https://doi.org/10.1090/S0025-5718-1975-0388967-5)
"""
@inline function flux_godunov(u_ll, u_rr, orientation::Integer,
                              equations::LinearizedEulerEquations2D)
    @unpack v_mean_global, rho_mean_global, c_mean_global = equations
    v1_mean = v_mean_global[1]
    v2_mean = v_mean_global[2]

    rho_prime_ll, v1_prime_ll, v2_prime_ll, p_prime_ll = u_ll
    rho_prime_rr, v1_prime_rr, v2_prime_rr, p_prime_rr = u_rr

    if orientation == 1
        # Eigenvalues of the flux matrix
        lambda1 = v1_mean
        lambda2 = v1_mean - c_mean_global
        lambda3 = v1_mean + c_mean_global

        lambda1_p = positive_part(lambda1)
        lambda2_p = positive_part(lambda2)
        lambda3_p = positive_part(lambda3)
        lambda2p3_half_p = 0.5f0 * (lambda2_p + lambda3_p)
        lambda3m2_half_p = 0.5f0 * (lambda3_p - lambda2_p)

        lambda1_m = negative_part(lambda1)
        lambda2_m = negative_part(lambda2)
        lambda3_m = negative_part(lambda3)
        lambda2p3_half_m = 0.5f0 * (lambda2_m + lambda3_m)
        lambda3m2_half_m = 0.5f0 * (lambda3_m - lambda2_m)

        f1p = (lambda1_p * rho_prime_ll +
               lambda3m2_half_p / c_mean_global * rho_mean_global * v1_prime_ll +
               (lambda2p3_half_p - lambda1_p) / c_mean_global^2 * p_prime_ll)
        f2p = (lambda2p3_half_p * v1_prime_ll +
               lambda3m2_half_p / c_mean_global * p_prime_ll / rho_mean_global)
        f3p = lambda1_p * v2_prime_ll
        f4p = (lambda3m2_half_p * c_mean_global * rho_mean_global * v1_prime_ll +
               lambda2p3_half_p * p_prime_ll)

        f1m = (lambda1_m * rho_prime_rr +
               lambda3m2_half_m / c_mean_global * rho_mean_global * v1_prime_rr +
               (lambda2p3_half_m - lambda1_m) / c_mean_global^2 * p_prime_rr)
        f2m = (lambda2p3_half_m * v1_prime_rr +
               lambda3m2_half_m / c_mean_global * p_prime_rr / rho_mean_global)
        f3m = lambda1_m * v2_prime_rr
        f4m = (lambda3m2_half_m * c_mean_global * rho_mean_global * v1_prime_rr +
               lambda2p3_half_m * p_prime_rr)

        f1 = f1p + f1m
        f2 = f2p + f2m
        f3 = f3p + f3m
        f4 = f4p + f4m
    else # orientation == 2
        # Eigenvalues of the flux matrix
        lambda1 = v2_mean
        lambda2 = v2_mean - c_mean_global
        lambda3 = v2_mean + c_mean_global

        lambda1_p = positive_part(lambda1)
        lambda2_p = positive_part(lambda2)
        lambda3_p = positive_part(lambda3)
        lambda2p3_half_p = 0.5f0 * (lambda2_p + lambda3_p)
        lambda3m2_half_p = 0.5f0 * (lambda3_p - lambda2_p)

        lambda1_m = negative_part(lambda1)
        lambda2_m = negative_part(lambda2)
        lambda3_m = negative_part(lambda3)
        lambda2p3_half_m = 0.5f0 * (lambda2_m + lambda3_m)
        lambda3m2_half_m = 0.5f0 * (lambda3_m - lambda2_m)

        f1p = (lambda1_p * rho_prime_ll +
               lambda3m2_half_p / c_mean_global * rho_mean_global * v2_prime_ll +
               (lambda2p3_half_p - lambda1_p) / c_mean_global^2 * p_prime_ll)
        f2p = lambda1_p * v1_prime_ll
        f3p = (lambda2p3_half_p * v2_prime_ll +
               lambda3m2_half_p / c_mean_global * p_prime_ll / rho_mean_global)
        f4p = (lambda3m2_half_p * c_mean_global * rho_mean_global * v2_prime_ll +
               lambda2p3_half_p * p_prime_ll)

        f1m = (lambda1_m * rho_prime_rr +
               lambda3m2_half_m / c_mean_global * rho_mean_global * v2_prime_rr +
               (lambda2p3_half_m - lambda1_m) / c_mean_global^2 * p_prime_rr)
        f2m = lambda1_m * v1_prime_rr
        f3m = (lambda2p3_half_m * v2_prime_rr +
               lambda3m2_half_m / c_mean_global * p_prime_rr / rho_mean_global)
        f4m = (lambda3m2_half_m * c_mean_global * rho_mean_global * v2_prime_rr +
               lambda2p3_half_m * p_prime_rr)

        f1 = f1p + f1m
        f2 = f2p + f2m
        f3 = f3p + f3m
        f4 = f4p + f4m
    end

    return SVector(f1, f2, f3, f4)
end

@inline function flux_godunov(u_ll, u_rr, normal_direction::AbstractVector,
                              equations::LinearizedEulerEquations2D)
    @unpack v_mean_global, rho_mean_global, c_mean_global = equations
    rho_prime_ll, v1_prime_ll, v2_prime_ll, p_prime_ll = u_ll
    rho_prime_rr, v1_prime_rr, v2_prime_rr, p_prime_rr = u_rr

    # Do not use `normalize` since we use `norm_` later to scale the eigenvalues
    norm_ = norm(normal_direction)
    normal_vector = normal_direction / norm_

    # Use normalized vector here, scaling is applied via eigenvalues of the flux matrix
    v_mean_normal = v_mean_global[1] * normal_vector[1] +
                    v_mean_global[2] * normal_vector[2]
    v_prime_normal_ll = v1_prime_ll * normal_vector[1] + v2_prime_ll * normal_vector[2]
    v_prime_normal_rr = v1_prime_rr * normal_vector[1] + v2_prime_rr * normal_vector[2]

    # Eigenvalues of the flux matrix
    lambda1 = v_mean_normal * norm_
    lambda2 = (v_mean_normal - c_mean_global) * norm_
    lambda3 = (v_mean_normal + c_mean_global) * norm_

    lambda1_p = positive_part(lambda1)
    lambda2_p = positive_part(lambda2)
    lambda3_p = positive_part(lambda3)
    lambda2p3_half_p = 0.5f0 * (lambda2_p + lambda3_p)
    lambda3m2_half_p = 0.5f0 * (lambda3_p - lambda2_p)

    lambda1_m = negative_part(lambda1)
    lambda2_m = negative_part(lambda2)
    lambda3_m = negative_part(lambda3)
    lambda2p3_half_m = 0.5f0 * (lambda2_m + lambda3_m)
    lambda3m2_half_m = 0.5f0 * (lambda3_m - lambda2_m)

    f1p = (lambda1_p * rho_prime_ll +
           lambda3m2_half_p / c_mean_global * rho_mean_global * v_prime_normal_ll +
           (lambda2p3_half_p - lambda1_p) / c_mean_global^2 * p_prime_ll)
    f2p = (((lambda1_p * normal_vector[2]^2 +
             lambda2p3_half_p * normal_vector[1]^2) * v1_prime_ll +
            (lambda2p3_half_p - lambda1_p) * prod(normal_vector) * v2_prime_ll) +
           lambda3m2_half_p / c_mean_global * normal_vector[1] * p_prime_ll /
           rho_mean_global)
    f3p = (((lambda1_p * normal_vector[1]^2 +
             lambda2p3_half_p * normal_vector[2]^2) * v2_prime_ll +
            (lambda2p3_half_p - lambda1_p) * prod(normal_vector) * v1_prime_ll) +
           lambda3m2_half_p / c_mean_global * normal_vector[2] * p_prime_ll /
           rho_mean_global)
    f4p = (lambda3m2_half_p * c_mean_global * rho_mean_global * v_prime_normal_ll +
           lambda2p3_half_p * p_prime_ll)

    f1m = (lambda1_m * rho_prime_rr +
           lambda3m2_half_m / c_mean_global * rho_mean_global * v_prime_normal_rr +
           (lambda2p3_half_m - lambda1_m) / c_mean_global^2 * p_prime_rr)
    f2m = (((lambda1_m * normal_vector[2]^2 +
             lambda2p3_half_m * normal_vector[1]^2) * v1_prime_rr +
            (lambda2p3_half_m - lambda1_m) * prod(normal_vector) * v2_prime_rr) +
           lambda3m2_half_m / c_mean_global * normal_vector[1] * p_prime_rr /
           rho_mean_global)
    f3m = (((lambda1_m * normal_vector[1]^2 +
             lambda2p3_half_m * normal_vector[2]^2) * v2_prime_rr +
            (lambda2p3_half_m - lambda1_m) * prod(normal_vector) * v1_prime_rr) +
           lambda3m2_half_m / c_mean_global * normal_vector[2] * p_prime_rr /
           rho_mean_global)
    f4m = (lambda3m2_half_m * c_mean_global * rho_mean_global * v_prime_normal_rr +
           lambda2p3_half_m * p_prime_rr)

    f1 = f1p + f1m
    f2 = f2p + f2m
    f3 = f3p + f3m
    f4 = f4p + f4m

    return SVector(f1, f2, f3, f4)
end

# Calculate estimate for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedEulerEquations2D)
    min_max_speed_davis(u_ll, u_rr, orientation, equations)
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::LinearizedEulerEquations2D)
    min_max_speed_davis(u_ll, u_rr, normal_direction, equations)
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedEulerEquations2D)
    @unpack v_mean_global, c_mean_global = equations

    λ_min = v_mean_global[orientation] - c_mean_global
    λ_max = v_mean_global[orientation] + c_mean_global

    return λ_min, λ_max
end

@inline function min_max_speed_davis(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::LinearizedEulerEquations2D)
    @unpack v_mean_global, c_mean_global = equations

    norm_ = norm(normal_direction)

    v_normal = v_mean_global[1] * normal_direction[1] +
               v_mean_global[2] * normal_direction[2]

    # The v_normals are already scaled by the norm
    λ_min = v_normal - c_mean_global * norm_
    λ_max = v_normal + c_mean_global * norm_

    return λ_min, λ_max
end

# Convert conservative variables to primitive
@inline cons2prim(u, equations::LinearizedEulerEquations2D) = u
@inline cons2entropy(u, ::LinearizedEulerEquations2D) = u
end # muladd
