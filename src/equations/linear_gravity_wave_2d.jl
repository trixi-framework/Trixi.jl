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
    \v_1' \\ v_2' \\ p' \\ b'
\end{pmatrix}
+
\partial_x
\begin{pmatrix}
    p \\ 0 \\ cs^2 v_1 \\ 0
\end{pmatrix}
+
\partial_z
\begin{pmatrix}
    0 \\ p \\ cs^2 v_2 \\ 0
\end{pmatrix}
=
\begin{pmatrix}
    0 \\ b \\ 0 \\ -N^2 v_2
\end{pmatrix}
```
The bar ``\bar{(\cdot)}`` indicates uniform mean flow variables and ``c`` is the speed of sound.
The unknowns are the acoustic velocities ``v' = (v_1', v_2')``, the pressure ``p'`` and the density ``\rho'``.
"""
struct LinearizedGravityWaveEquations2D{RealT <: Real} <:
       AbstractLinearizedEulerEquations{2, 4}
    cs::RealT  # speed of sound
    fb::RealT # Buoyancy frequency
end

function LinearizedGravityWaveEquations2D(cs::Real, fb::Real)
    return LinearizedGravityWaveEquations2D(cs, fb)
end

function varnames(::typeof(cons2cons), ::LinearizedGravityWaveEquations2D)
    ("u", "w", "p", "b")
end
function varnames(::typeof(cons2prim), ::LinearizedGravityWaveEquations2D)
    ("u", "w", "p", "b")
end

"""
    initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations2D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t,
                                            equations::LinearizedGravityWaveEquations2D)
    A = 5000
    H = 10000
    b0 = 0.01
    xc = 0.0
    binv = (1 + (x[1] - xc)^2 / A^2)
    b = b0 * sin(pi * x[2] / H) / binv

    return SVector(0.0, 0.0, 0.0, b)
end

"""
boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
surface_flux_function, equations::ShallowWaterEquations2D)

Should be used together with [`TreeMesh`](@ref).
"""


@inline function boundary_condition_slip_wall(u_inner, orientation,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::LinearizedGravityWaveEquations2D)
    ## get the appropriate normal vector from the orientation
    if orientation == 1
        u_boundary = SVector(-u_inner[1], u_inner[2], u_inner[3], u_inner[4])
    else # orientation == 2
        u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4])
    end

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        println(surface_flux_function)
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
        
    return flux
end

@inline function source_terms_convergence_test(u, x, t,
                              equations::LinearizedGravityWaveEquations2D)
    # Same settings as in `initial_condition`
    @unpack fb = equations
    _, v2, _, b = u
    du1 = 0.0
    du2 = b
    du3 = 0
    du4 = -fb^2 * v2

    return SVector(du1, du2, du3, du4)
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer,
                      equations::LinearizedGravityWaveEquations2D)
    @unpack cs = equations
    v1, v2, p, b = u
    if orientation == 1
        f1 = p
        f2 = 0.0
        f3 = cs^2 * v1
        f4 = 0.0
    else
        f1 = 0.0
        f2 = p
        f3 = cs^2 * v2
        f4 = 0.0
    end

    return SVector(f1, f2, f3, f4)
end

# Calculate 1D flux for a single point
@inline function flux(u, normal_direction::AbstractVector,
                      equations::LinearizedGravityWaveEquations2D)
    @unpack v_mean_global, c_mean_global, rho_mean_global = equations
    v1, v2, p, b = u

    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]

    f1 = normal_direction[1] * p
    f2 = normal_direction[2] * p
    f3 = v_normal * cs^2
    f4 = 0.0

    return SVector(f1, f2, f3, f4)
end

@inline have_constant_speed(::LinearizedGravityWaveEquations2D) = True()

@inline function max_abs_speeds(equations::LinearizedGravityWaveEquations2D)
    @unpack cs = equations
    return cs, cs
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedGravityWaveEquations2D)
    @unpack v_mean_global, c_mean_global = equations
    if orientation == 1
        return abs(v_mean_global[1]) + c_mean_global
    else # orientation == 2
        return abs(v_mean_global[2]) + c_mean_global
    end
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::LinearizedGravityWaveEquations2D)
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
@inline function flux_lmars2(u_ll, u_rr, orientation::Integer,
                              equations::LinearizedGravityWaveEquations2D)
    @unpack cs = equations

    v1_ll, v2_ll, p_ll, b_ll = u_ll
    v1_rr, v2_rr, p_rr, b_rr = u_rr

    if orientation == 1
        f1 = (p_ll + p_rr) * 0.5 - 0.5*cs*(v1_rr - v1_ll)
        f2 = 0.0
        f3 = cs^2 *( (v1_ll + v1_rr) * 0.5 - 1/(2*cs)*(p_rr - p_ll))
        f4 = 0.0
        println("a volte qui")
    else # orientation == 2
        f1 = 0.0
        f2 = (p_ll + p_rr) * 0.5 - 0.5*cs*(v2_rr - v2_ll)
        f3 = cs^2 *( (v2_ll + v2_rr) * 0.5 - 1/(2*cs)*(p_rr - p_ll))
        f4 = 0.0
        println("sono anche qui")
    end
    println("sono qui")
    return SVector(f1, f2, f3, f4)
end

@inline function flux_lmars2(u_ll, u_rr, normal_direction::AbstractVector,
                              equations::LinearizedGravityWaveEquations2D)
    @unpack cs = equations
    v1_ll, v2_ll, p_ll, b_ll = u_ll
    v1_rr, v2_rr, p_rr, b_rr = u_rr

    norm_ = norm(normal_direction)
    normal_vector = normal_direction / norm_

    v_normal_ll = v1_ll * normal_vector[1] + v2_ll * normal_vector[2]
    v_normal_rr = v1_rr * normal_vector[1] + v2_rr * normal_vector[2]

    f1 = normal_vector[1] * (0.5 * (p_ll + p_rr) - 0.5*cs*(v_normal_rr - v_normal_ll))
    f2 = normal_vector[2] * (0.5 * (p_ll + p_rr) - 0.5*cs*(v_normal_rr - v_normal_ll))
    f3 = (0.5 * (v_normal_ll + v_normal_rr) - 1/(2*cs)*(p_rr - p_ll) )* cs^2
    f4 = 0.0

    return SVector(f1, f2, f3, f4)
end

@inline function flux_godunov(u_ll, u_rr, orientation::Integer,
    equations::LinearizedGravityWaveEquations2D)
@unpack cs = equations

v1_ll, v2_ll, p_ll, b_ll = u_ll
v1_rr, v2_rr, p_rr, b_rr = u_rr

if orientation == 1
f1 = (p_ll + p_rr) * 0.5
f2 = 0.0
f3 = cs^2 * (v1_ll + v1_rr) * 0.5
f4 = 0.0
else # orientation == 2
f1 = 0.0
f2 = (p_ll + p_rr) * 0.5
f3 = cs^2 * (v2_ll + v2_rr) * 0.5
f4 = 0.0
end

return SVector(f1, f2, f3, f4)
end

@inline function flux_lmars(u_ll, u_rr, normal_direction::AbstractVector,
    equations::LinearizedGravityWaveEquations2D)
@unpack cs = equations
v1_ll, v2_ll, p_ll, b_ll = u_ll
v1_rr, v2_rr, p_rr, b_rr = u_rr

norm_ = norm(normal_direction)
normal_vector = normal_direction / norm_

v_normal_ll = v1_ll * normal_vector[1] + v2_ll * normal_vector[2]
v_normal_rr = v1_rr * normal_vector[1] + v2_rr * normal_vector[2]

f1 = normal_vector[1] * 0.5 * (p_ll + p_rr)
f2 = normal_vector[2] * 0.5 * (p_ll + p_rr)
f3 = 0.5 * (v_normal_ll + v_normal_rr) * cs^2
f4 = 0.0

return SVector(f1, f2, f3, f4)
end

# Calculate estimate for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedGravityWaveEquations2D)
    min_max_speed_davis(u_ll, u_rr, orientation, equations)
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::LinearizedGravityWaveEquations2D)
    min_max_speed_davis(u_ll, u_rr, normal_direction, equations)
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedGravityWaveEquations2D)
    @unpack v_mean_global, c_mean_global = equations

    λ_min = v_mean_global[orientation] - c_mean_global
    λ_max = v_mean_global[orientation] + c_mean_global

    return λ_min, λ_max
end

@inline function min_max_speed_davis(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::LinearizedGravityWaveEquations2D)
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
@inline cons2prim(u, equations::LinearizedGravityWaveEquations2D) = u
@inline cons2entropy(u, ::LinearizedGravityWaveEquations2D) = u
end # muladd
