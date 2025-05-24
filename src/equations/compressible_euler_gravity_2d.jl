# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent
@doc raw"""
    CompressibleEulerEquationsWithGravity2D(gamma)

The compressible Euler equations with gravity,
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
\rho \\ \rho v_1 \\ \rho v_2 \\ \rho e \\ \Phi
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
    \rho v_1 \\ \rho v_1^2 + p \\ \rho v_1 v_2 \\ (\rho e +p) v_1 \\ 0
\end{pmatrix}
+
\frac{\partial}{\partial y}
\begin{pmatrix}
\rho v_2 \\ \rho v_1 v_2 \\ \rho v_2^2 + p \\ (\rho e +p) v_2 \\ 0
\end{pmatrix}
=
\begin{pmatrix}
0 \\ - \rho \nabla \Phi \\ 0 \\ 0
\end{pmatrix}
```
for an ideal gas with ratio of specific heat `gamma` in two space dimensions.

Here, ``\rho`` is the density, ``v_1``,`v_2` the velocities, ``e`` the specific total energy, 
which includes the internal, kinetik, and geopotential energy, `\Phi` is the gravitational 
geopotential, and
```math
p = (\gamma - 1) \left( \rho e - \frac{1}{2} \rho (v_1^2+v_2^2) - \rho \Phi \right)
```
the pressure.

References:
- Souza, A. N., He, J., Bischoff, T., Waruszewski, M., Novak, L., Barra, V., ... & Schneider, T. (2023). The flux‐differencing discontinuous galerkin method applied to an idealized fully compressible nonhydrostatic dry atmosphere. Journal of Advances in Modeling Earth Systems, 15(4), e2022MS003527. https://doi.org/10.1029/2022MS003527.
- Waruszewski, M., Kozdon, J. E., Wilcox, L. C., Gibson, T. H., & Giraldo, F. X. (2022). Entropy stable discontinuous Galerkin methods for balance laws in non-conservative form: Applications to the Euler equations with gravity. Journal of Computational Physics, 468, 111507. https://doi.org/10.1016/j.jcp.2022.111507.
"""
struct CompressibleEulerEquationsWithGravity2D{RealT <: Real} <:
       AbstractCompressibleEulerEquations{2, 5}
    gamma::RealT               # ratio of specific heats
    inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications

    function CompressibleEulerEquationsWithGravity2D(gamma)
        γ, inv_gamma_minus_one = promote(gamma, inv(gamma - 1))
        new{typeof(γ)}(γ, inv_gamma_minus_one)
    end
end

have_nonconservative_terms(::CompressibleEulerEquationsWithGravity2D) = True()
varnames(::typeof(cons2cons), ::CompressibleEulerEquationsWithGravity2D) = ("rho",
                                                                            "rho_v1",
                                                                            "rho_v2",
                                                                            "rho_e",
                                                                            "phi")
varnames(::typeof(cons2prim), ::CompressibleEulerEquationsWithGravity2D) = ("rho",
                                                                            "v1",
                                                                            "v2",
                                                                            "p",
                                                                            "phi")

# Set initial conditions at physical location `x` for time `t`
# """
#     initial_condition_constant(x, t, equations::CompressibleEulerEquationsWithGravity2D)

# A constant initial condition to test free-stream preservation.
# """
# function initial_condition_constant(x, t, equations::CompressibleEulerEquationsWithGravity2D)
#   rho = 1.0
#   rho_v1 = 0.1
#   rho_v2 = -0.2
#   rho_e = 10.0
#   return SVector(rho, rho_v1, rho_v2, rho_e)
# end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                    equations::CompressibleEulerEquationsWithGravity2D)

Determine the boundary numerical surface flux for a slip wall condition.
Imposes a zero normal velocity at the wall.
Density is taken from the internal solution state and pressure is computed as an
exact solution of a 1D Riemann problem. Further details about this boundary state
are available in the paper:
- J. J. W. van der Vegt and H. van der Ven (2002)
    Slip flow boundary conditions in discontinuous Galerkin discretizations of
    the Euler equations of gas dynamics
    [PDF](https://reports.nlr.nl/bitstream/handle/10921/692/TP-2002-300.pdf?sequence=1)

Details about the 1D pressure Riemann solution can be found in Section 6.3.3 of the book
- Eleuterio F. Toro (2009)
    Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
    3rd edition
    [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)

Should be used together with [`UnstructuredMesh2D`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner,
                                              normal_direction::AbstractVector,
                                              x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquationsWithGravity2D)
    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal = normal_direction / norm_

    # rotate the internal solution state
    u_local = rotate_to_x(u_inner, normal, equations)

    # compute the primitive variables
    rho_local, v_normal, v_tangent, p_local, _ = cons2prim(u_local, equations)

    # Get the solution of the pressure Riemann problem
    # See Section 6.3.3 of
    # Eleuterio F. Toro (2009)
    # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
    # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
    if v_normal <= 0.0
        sound_speed = sqrt(equations.gamma * p_local / rho_local) # local sound speed
        p_star = p_local *
                 (1 + 0.5 * (equations.gamma - 1) * v_normal / sound_speed)^(2 *
                                                                             equations.gamma *
                                                                             equations.inv_gamma_minus_one)
    else # v_normal > 0.0
        A = 2 / ((equations.gamma + 1) * rho_local)
        B = p_local * (equations.gamma - 1) / (equations.gamma + 1)
        p_star = p_local +
                 0.5 * v_normal / A *
                 (v_normal + sqrt(v_normal^2 + 4 * A * (p_local + B)))
    end

    # For the slip wall we directly set the flux as the normal velocity is zero
    return (SVector(zero(eltype(u_inner)),
                    p_star * normal[1],
                    p_star * normal[2],
                    zero(eltype(u_inner)),
                    zero(eltype(u_inner))) * norm_,
            SVector(zero(eltype(u_inner)),
                    zero(eltype(u_inner)),
                    zero(eltype(u_inner)),
                    zero(eltype(u_inner)),
                    zero(eltype(u_inner))) * norm_)
end

"""
    boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
                                    surface_flux_function, equations::CompressibleEulerEquationsWithGravity2D)

Should be used together with [`TreeMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, orientation,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquationsWithGravity2D)
    # get the appropriate normal vector from the orientation
    if orientation == 1
        normal_direction = SVector(1, 0)
    else # orientation == 2
        normal_direction = SVector(0, 1)
    end

    # compute and return the flux using `boundary_condition_slip_wall` routine above
    return boundary_condition_slip_wall(u_inner, normal_direction, direction,
                                        x, t, surface_flux_function, equations)
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, direction, x, t,
                                    surface_flux_function, equations::CompressibleEulerEquationsWithGravity2D)

Should be used together with [`StructuredMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner,
                                              normal_direction::AbstractVector,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquationsWithGravity2D)
    # flip sign of normal to make it outward pointing, then flip the sign of the normal flux back
    # to be inward pointing on the -x and -y sides due to the orientation convention used by StructuredMesh
    if isodd(direction)
        fluxes = boundary_condition_slip_wall(u_inner, -normal_direction,
                                              x, t, surface_flux_function, equations)
        boundary_flux = (-fluxes[1], -fluxes[2])
    else
        boundary_flux = boundary_condition_slip_wall(u_inner, normal_direction,
                                                     x, t, surface_flux_function,
                                                     equations)
    end

    return boundary_flux
end

# Calculate 2D flux for a single point
@inline function flux(u, orientation::Integer,
                      equations::CompressibleEulerEquationsWithGravity2D)
    rho, rho_v1, rho_v2, rho_e, phi = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2) - rho * phi)
    if orientation == 1
        f1 = rho_v1
        f2 = rho_v1 * v1 + p
        f3 = rho_v1 * v2
        f4 = (rho_e + p) * v1
    else
        f1 = rho_v2
        f2 = rho_v2 * v1
        f3 = rho_v2 * v2 + p
        f4 = (rho_e + p) * v2
    end
    return SVector(f1, f2, f3, f4, zero(eltype(u)))
end

# Calculate 2D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector,
                      equations::CompressibleEulerEquationsWithGravity2D)
    rho_e = u[4]
    rho, v1, v2, p, _ = cons2prim(u, equations)

    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
    rho_v_normal = rho * v_normal
    f1 = rho_v_normal
    f2 = rho_v_normal * v1 + p * normal_direction[1]
    f3 = rho_v_normal * v2 + p * normal_direction[2]
    f4 = (rho_e + p) * v_normal
    return SVector(f1, f2, f3, f4, zero(eltype(u)))
end

"""
    flux_shima_etal(u_ll, u_rr, orientation_or_normal_direction,
                    equations::CompressibleEulerEquationsWithGravity2D)

This flux is is a modification of the original kinetic energy preserving two-point flux by
- Yuichi Kuya, Kosuke Totani and Soshi Kawai (2018)
    Kinetic energy and entropy preserving schemes for compressible flows
    by split convective forms
    [DOI: 10.1016/j.jcp.2018.08.058](https://doi.org/10.1016/j.jcp.2018.08.058)

The modification is in the energy flux to guarantee pressure equilibrium and was developed by
- Nao Shima, Yuichi Kuya, Yoshiharu Tamaki, Soshi Kawai (JCP 2020)
    Preventing spurious pressure oscillations in split convective form discretizations for
    compressible flows
    [DOI: 10.1016/j.jcp.2020.110060](https://doi.org/10.1016/j.jcp.2020.110060)
"""
@inline function flux_shima_etal(u_ll, u_rr, orientation::Integer,
                                 equations::CompressibleEulerEquationsWithGravity2D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)

    # Average each factor of products in flux
    rho_avg = 1 / 2 * (rho_ll + rho_rr)
    v1_avg = 1 / 2 * (v1_ll + v1_rr)
    v2_avg = 1 / 2 * (v2_ll + v2_rr)
    p_avg = 1 / 2 * (p_ll + p_rr)
    kin_avg = 1 / 2 * (v1_ll * v1_rr + v2_ll * v2_rr)

    # Calculate fluxes depending on orientation
    if orientation == 1
        pv1_avg = 1 / 2 * (p_ll * v1_rr + p_rr * v1_ll)
        f1 = rho_avg * v1_avg
        f2 = f1 * v1_avg + p_avg
        f3 = f1 * v2_avg
        f4 = p_avg * v1_avg * equations.inv_gamma_minus_one + f1 * kin_avg + pv1_avg
    else
        pv2_avg = 1 / 2 * (p_ll * v2_rr + p_rr * v2_ll)
        f1 = rho_avg * v2_avg
        f2 = f1 * v1_avg
        f3 = f1 * v2_avg + p_avg
        f4 = p_avg * v2_avg * equations.inv_gamma_minus_one + f1 * kin_avg + pv2_avg
    end

    return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end

@inline function flux_shima_etal(u_ll, u_rr, normal_direction::AbstractVector,
                                 equations::CompressibleEulerEquationsWithGravity2D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)
    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    # Average each factor of products in flux
    rho_avg = 1 / 2 * (rho_ll + rho_rr)
    v1_avg = 1 / 2 * (v1_ll + v1_rr)
    v2_avg = 1 / 2 * (v2_ll + v2_rr)
    v_dot_n_avg = 1 / 2 * (v_dot_n_ll + v_dot_n_rr)
    p_avg = 1 / 2 * (p_ll + p_rr)
    velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_avg * v_dot_n_avg
    f2 = f1 * v1_avg + p_avg * normal_direction[1]
    f3 = f1 * v2_avg + p_avg * normal_direction[2]
    f4 = (f1 * velocity_square_avg +
          p_avg * v_dot_n_avg * equations.inv_gamma_minus_one
          + 0.5 * (p_ll * v_dot_n_rr + p_rr * v_dot_n_ll))

    return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end

"""
    flux_kennedy_gruber(u_ll, u_rr, orientation_or_normal_direction,
                        equations::CompressibleEulerEquationsWithGravity2D)

Kinetic energy preserving two-point flux by
- Kennedy and Gruber (2008)
    Reduced aliasing formulations of the convective terms within the
    Navier-Stokes equations for a compressible fluid
    [DOI: 10.1016/j.jcp.2007.09.020](https://doi.org/10.1016/j.jcp.2007.09.020)
"""
@inline function flux_kennedy_gruber(u_ll, u_rr, orientation::Integer,
                                     equations::CompressibleEulerEquationsWithGravity2D)
    # Unpack left and right state
    rho_e_ll = u_ll[4]
    rho_e_rr = u_rr[4]
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)

    # Average each factor of products in flux
    rho_avg = 1 / 2 * (rho_ll + rho_rr)
    v1_avg = 1 / 2 * (v1_ll + v1_rr)
    v2_avg = 1 / 2 * (v2_ll + v2_rr)
    p_avg = 1 / 2 * (p_ll + p_rr)
    e_avg = 1 / 2 * (rho_e_ll / rho_ll + rho_e_rr / rho_rr)

    # Calculate fluxes depending on orientation
    if orientation == 1
        f1 = rho_avg * v1_avg
        f2 = rho_avg * v1_avg * v1_avg + p_avg
        f3 = rho_avg * v1_avg * v2_avg
        f4 = (rho_avg * e_avg + p_avg) * v1_avg
    else
        f1 = rho_avg * v2_avg
        f2 = rho_avg * v2_avg * v1_avg
        f3 = rho_avg * v2_avg * v2_avg + p_avg
        f4 = (rho_avg * e_avg + p_avg) * v2_avg
    end

    return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end

@inline function flux_kennedy_gruber(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::CompressibleEulerEquationsWithGravity2D)
    # Unpack left and right state
    rho_e_ll = u_ll[4]
    rho_e_rr = u_rr[4]
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)

    # Average each factor of products in flux
    rho_avg = 0.5 * (rho_ll + rho_rr)
    v1_avg = 0.5 * (v1_ll + v1_rr)
    v2_avg = 0.5 * (v2_ll + v2_rr)
    v_dot_n_avg = v1_avg * normal_direction[1] + v2_avg * normal_direction[2]
    p_avg = 0.5 * (p_ll + p_rr)
    e_avg = 0.5 * (rho_e_ll / rho_ll + rho_e_rr / rho_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_avg * v_dot_n_avg
    f2 = f1 * v1_avg + p_avg * normal_direction[1]
    f3 = f1 * v2_avg + p_avg * normal_direction[2]
    f4 = f1 * e_avg + p_avg * v_dot_n_avg

    return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end

"""
    flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleEulerEquationsWithGravity2D)

Entropy conserving two-point flux by
- Chandrashekar (2013)
    Kinetic Energy Preserving and Entropy Stable Finite Volume Schemes
    for Compressible Euler and Navier-Stokes Equations
    [DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation::Integer,
                                    equations::CompressibleEulerEquationsWithGravity2D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)
    beta_ll = 0.5 * rho_ll / p_ll
    beta_rr = 0.5 * rho_rr / p_rr
    specific_kin_ll = 0.5 * (v1_ll^2 + v2_ll^2)
    specific_kin_rr = 0.5 * (v1_rr^2 + v2_rr^2)

    # Compute the necessary mean values
    rho_avg = 0.5 * (rho_ll + rho_rr)
    rho_mean = ln_mean(rho_ll, rho_rr)
    beta_mean = ln_mean(beta_ll, beta_rr)
    beta_avg = 0.5 * (beta_ll + beta_rr)
    v1_avg = 0.5 * (v1_ll + v1_rr)
    v2_avg = 0.5 * (v2_ll + v2_rr)
    p_mean = 0.5 * rho_avg / beta_avg
    velocity_square_avg = specific_kin_ll + specific_kin_rr

    # Calculate fluxes depending on orientation
    if orientation == 1
        f1 = rho_mean * v1_avg
        f2 = f1 * v1_avg + p_mean
        f3 = f1 * v2_avg
        f4 = f1 * 0.5 * (1 / (equations.gamma - 1) / beta_mean - velocity_square_avg) +
             f2 * v1_avg + f3 * v2_avg
    else
        f1 = rho_mean * v2_avg
        f2 = f1 * v1_avg
        f3 = f1 * v2_avg + p_mean
        f4 = f1 * 0.5 * (1 / (equations.gamma - 1) / beta_mean - velocity_square_avg) +
             f2 * v1_avg + f3 * v2_avg
    end

    return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end

"""
    flux_ranocha(u_ll, u_rr, orientation_or_normal_direction,
                    equations::CompressibleEulerEquationsWithGravity2D)

Entropy conserving and kinetic energy preserving two-point flux by
- Hendrik Ranocha (2018)
    Generalised Summation-by-Parts Operators and Entropy Stability of Numerical Methods
    for Hyperbolic Balance Laws
    [PhD thesis, TU Braunschweig](https://cuvillier.de/en/shop/publications/7743)
See also
- Hendrik Ranocha (2020)
    Entropy Conserving and Kinetic Energy Preserving Numerical Methods for
    the Euler Equations Using Summation-by-Parts Operators
    [Proceedings of ICOSAHOM 2018](https://doi.org/10.1007/978-3-030-39647-3_42)
"""
@inline function flux_ranocha(u_ll, u_rr, orientation::Integer,
                              equations::CompressibleEulerEquationsWithGravity2D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)

    # Compute the necessary mean values
    rho_mean = ln_mean(rho_ll, rho_rr)
    # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
    # in exact arithmetic since
    #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
    #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
    inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
    v1_avg = 0.5 * (v1_ll + v1_rr)
    v2_avg = 0.5 * (v2_ll + v2_rr)
    p_avg = 0.5 * (p_ll + p_rr)
    velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr)

    # Calculate fluxes depending on orientation
    if orientation == 1
        f1 = rho_mean * v1_avg
        f2 = f1 * v1_avg + p_avg
        f3 = f1 * v2_avg
        f4 = f1 *
             (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one) +
             0.5 * (p_ll * v1_rr + p_rr * v1_ll)
    else
        f1 = rho_mean * v2_avg
        f2 = f1 * v1_avg
        f3 = f1 * v2_avg + p_avg
        f4 = f1 *
             (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one) +
             0.5 * (p_ll * v2_rr + p_rr * v2_ll)
    end

    return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end

@inline function flux_ranocha(u_ll, u_rr, normal_direction::AbstractVector,
                              equations::CompressibleEulerEquationsWithGravity2D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)
    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    # Compute the necessary mean values
    rho_mean = ln_mean(rho_ll, rho_rr)
    # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
    # in exact arithmetic since
    #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
    #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
    inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
    v1_avg = 0.5 * (v1_ll + v1_rr)
    v2_avg = 0.5 * (v2_ll + v2_rr)
    p_avg = 0.5 * (p_ll + p_rr)
    velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_mean * 0.5 * (v_dot_n_ll + v_dot_n_rr)
    f2 = f1 * v1_avg + p_avg * normal_direction[1]
    f3 = f1 * v2_avg + p_avg * normal_direction[2]
    f4 = (f1 * (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one)
          +
          0.5 * (p_ll * v_dot_n_rr + p_rr * v_dot_n_ll))

    return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end

function flux_nonconservative_waruszewski(u_ll, u_rr, normal_direction::AbstractVector,
                                          equations::CompressibleEulerEquationsWithGravity2D)
    rho_ll, _, _, _, phi_ll = u_ll
    rho_rr, _, _, _, phi_rr = u_rr

    # We omit the 0.5 in the density average since jl always multiplies the non-conservative flux with 0.5
    noncons = ln_mean(rho_ll, rho_rr) * (phi_rr - phi_ll)
    # noncons = 0.5 * (rho_ll + rho_rr) * (phi_rr - phi_ll)

    f0 = zero(eltype(u_ll))
    return SVector(f0, noncons * normal_direction[1], noncons * normal_direction[2],
                   f0, f0)
end

function flux_nonconservative_waruszewski(u_ll, u_rr, orientation::Integer,
                                          equations::CompressibleEulerEquationsWithGravity2D)
    rho_ll, _, _, _, phi_ll = u_ll
    rho_rr, _, _, _, phi_rr = u_rr

    # We omit the 0.5 in the density average since jl always multiplies the non-conservative flux with 0.5
    noncons = ln_mean(rho_ll, rho_rr) * (phi_rr - phi_ll)
    # noncons = 0.5 * (rho_ll + rho_rr) * (phi_rr - phi_ll)

    f0 = zero(eltype(u_ll))
    if orientation == 1
        return SVector(f0, noncons, f0, f0, f0)
    else #if orientation == 2
        return SVector(f0, f0, noncons, f0, f0)
    end
end

"""
    FluxLMARS(c)(u_ll, u_rr, orientation_or_normal_direction,
                    equations::CompressibleEulerEquationsWithGravity2D)

Low Mach number approximate Riemann solver (LMARS) for atmospheric flows using
an estimate `c` of the speed of sound.

References:
- Xi Chen et al. (2013)
    A Control-Volume Model of the Compressible Euler Equations with a Vertical
    Lagrangian Coordinate
    [DOI: 10.1175/MWR-D-12-00129.1](https://doi.org/10.1175/mwr-d-12-00129.1)
"""
# The struct is already defined in CompressibleEulerEquations2D

@inline function (flux_lmars::FluxLMARS)(u_ll, u_rr, orientation::Integer,
                                         equations::CompressibleEulerEquationsWithGravity2D)
    c = flux_lmars.speed_of_sound

    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)

    if orientation == 1
        v_ll = v1_ll
        v_rr = v1_rr
    else # orientation == 2
        v_ll = v2_ll
        v_rr = v2_rr
    end

    rho = 0.5f0 * (rho_ll + rho_rr)
    p = 0.5f0 * (p_ll + p_rr) - 0.5f0 * c * rho * (v_rr - v_ll)
    v = 0.5f0 * (v_ll + v_rr) - 1 / (2 * c * rho) * (p_rr - p_ll)

    # We treat the energy term analogous to the potential temperature term in the paper by
    # Chen et al., i.e. we use p_ll and p_rr, and not p
    if v >= 0
        f1, f2, f3, f4, _ = v * u_ll
        f4 = f4 + p_ll * v
    else
        f1, f2, f3, f4, _ = v * u_rr
        f4 = f4 + p_rr * v
    end

    if orientation == 1
        f2 = f2 + p
    else # orientation == 2
        f3 = f3 + p
    end

    return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end

@inline function (flux_lmars::FluxLMARS)(u_ll, u_rr, normal_direction::AbstractVector,
                                         equations::CompressibleEulerEquationsWithGravity2D)
    c = flux_lmars.speed_of_sound

    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)

    v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    # Note that this is the same as computing v_ll and v_rr with a normalized normal vector
    # and then multiplying v by `norm_` again, but this version is slightly faster.
    norm_ = norm(normal_direction)

    rho = 0.5f0 * (rho_ll + rho_rr)
    p = 0.5f0 * (p_ll + p_rr) - 0.5f0 * c * rho * (v_rr - v_ll) / norm_
    v = 0.5f0 * (v_ll + v_rr) - 1 / (2 * c * rho) * (p_rr - p_ll) * norm_

    # We treat the energy term analogous to the potential temperature term in the paper by
    # Chen et al., i.e. we use p_ll and p_rr, and not p
    if v >= 0
        f1, f2, f3, f4, _ = u_ll * v
        f4 = f4 + p_ll * v
    else
        f1, f2, f3, f4, _ = u_rr * v
        f4 = f4 + p_rr * v
    end

    return SVector(f1,
                   f2 + p * normal_direction[1],
                   f3 + p * normal_direction[2],
                   f4, zero(eltype(u_ll)))
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::CompressibleEulerEquationsWithGravity2D)
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)

    # Get the velocity value in the appropriate direction
    if orientation == 1
        v_ll = v1_ll
        v_rr = v1_rr
    else # orientation == 2
        v_ll = v2_ll
        v_rr = v2_rr
    end
    # Calculate sound speeds
    c_ll = sqrt(equations.gamma * p_ll / rho_ll)
    c_rr = sqrt(equations.gamma * p_rr / rho_rr)

    λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::CompressibleEulerEquationsWithGravity2D)
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)

    # Calculate normal velocities and sound speed
    # left
    v_ll = (v1_ll * normal_direction[1]
            +
            v2_ll * normal_direction[2])
    c_ll = sqrt(equations.gamma * p_ll / rho_ll)
    # right
    v_rr = (v1_rr * normal_direction[1]
            +
            v2_rr * normal_direction[2])
    c_rr = sqrt(equations.gamma * p_rr / rho_rr)

    return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr) * norm(normal_direction)
end

# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::CompressibleEulerEquationsWithGravity2D)
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)

    if orientation == 1 # x-direction
        λ_min = v1_ll - sqrt(equations.gamma * p_ll / rho_ll)
        λ_max = v1_rr + sqrt(equations.gamma * p_rr / rho_rr)
    else # y-direction
        λ_min = v2_ll - sqrt(equations.gamma * p_ll / rho_ll)
        λ_max = v2_rr + sqrt(equations.gamma * p_rr / rho_rr)
    end

    return λ_min, λ_max
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::CompressibleEulerEquationsWithGravity2D)
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)

    v_normal_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_normal_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    norm_ = norm(normal_direction)
    # The v_normals are already scaled by the norm
    λ_min = v_normal_ll - sqrt(equations.gamma * p_ll / rho_ll) * norm_
    λ_max = v_normal_rr + sqrt(equations.gamma * p_rr / rho_rr) * norm_

    return λ_min, λ_max
end

@inline function splitting_drikakis_tsangaris(u, orientation_or_normal_direction,
                                              equations::CompressibleEulerEquationsWithGravity2D)
    fm = splitting_drikakis_tsangaris(u, Val{:minus}(), orientation_or_normal_direction,
                                      equations)
    fp = splitting_drikakis_tsangaris(u, Val{:plus}(), orientation_or_normal_direction,
                                      equations)
    return fm, fp
end

@inline function splitting_drikakis_tsangaris(u, ::Val{:plus},
                                              normal_direction::AbstractVector,
                                              equations::CompressibleEulerEquationsWithGravity2D)
    rho, v1, v2, p, _ = cons2prim(u, equations)
    a = sqrt(equations.gamma * p / rho)
    H = (u[4] + p) / rho

    v_n = normal_direction[1] * v1 + normal_direction[2] * v2

    lambda1 = v_n + a
    lambda2 = v_n - a

    lambda1_p = positive_part(lambda1) # Same as (lambda_i + abs(lambda_i)) / 2, but faster :)
    lambda2_p = positive_part(lambda2)

    rhoa_2gamma = 0.5f0 * rho * a / equations.gamma
    f1p = 0.5f0 * rho * (lambda1_p + lambda2_p)
    f2p = f1p * v1 + rhoa_2gamma * normal_direction[1] * (lambda1_p - lambda2_p)
    f3p = f1p * v2 + rhoa_2gamma * normal_direction[2] * (lambda1_p - lambda2_p)
    f4p = f1p * H

    return SVector(f1p, f2p, f3p, f4p, zero(eltype(u)))
end

@inline function splitting_drikakis_tsangaris(u, ::Val{:minus},
                                              normal_direction::AbstractVector,
                                              equations::CompressibleEulerEquationsWithGravity2D)
    rho, v1, v2, p, _ = cons2prim(u, equations)
    a = sqrt(equations.gamma * p / rho)
    H = (u[4] + p) / rho

    v_n = normal_direction[1] * v1 + normal_direction[2] * v2

    lambda1 = v_n + a
    lambda2 = v_n - a

    lambda1_m = negative_part(lambda1) # Same as (lambda_i - abs(lambda_i)) / 2, but faster :)
    lambda2_m = negative_part(lambda2)

    rhoa_2gamma = 0.5f0 * rho * a / equations.gamma
    f1m = 0.5f0 * rho * (lambda1_m + lambda2_m)
    f2m = f1m * v1 + rhoa_2gamma * normal_direction[1] * (lambda1_m - lambda2_m)
    f3m = f1m * v2 + rhoa_2gamma * normal_direction[2] * (lambda1_m - lambda2_m)
    f4m = f1m * H

    return SVector(f1m, f2m, f3m, f4m, zero(eltype(u)))
end

# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this rotation of the state vector
@inline function rotate_to_x(u, normal_vector,
                             equations::CompressibleEulerEquationsWithGravity2D)
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D rotation matrix with normal and tangent directions of the form
    # [ 1    0    0   0;
    #   0   n_1  n_2  0;
    #   0   t_1  t_2  0;
    #   0    0    0   1 ]
    # where t_1 = -n_2 and t_2 = n_1

    return SVector(u[1],
                   c * u[2] + s * u[3],
                   -s * u[2] + c * u[3],
                   u[4],
                   u[5])
end

# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this back-rotation of the state vector
@inline function rotate_from_x(u, normal_vector,
                               equations::CompressibleEulerEquationsWithGravity2D)
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D back-rotation matrix with normal and tangent directions of the form
    # [ 1    0    0   0;
    #   0   n_1  t_1  0;
    #   0   n_2  t_2  0;
    #   0    0    0   1 ]
    # where t_1 = -n_2 and t_2 = n_1

    return SVector(u[1],
                   c * u[2] - s * u[3],
                   s * u[2] + c * u[3],
                   u[4],
                   u[5])
end

"""
    flux_hllc(u_ll, u_rr, orientation, equations::CompressibleEulerEquationsWithGravity2D)

Computes the HLLC flux (HLL with Contact) for compressible Euler equations developed by E.F. Toro
[Lecture slides](http://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf)
Signal speeds: [DOI: 10.1137/S1064827593260140](https://doi.org/10.1137/S1064827593260140)
"""
function flux_hllc(u_ll, u_rr, orientation::Integer,
                   equations::CompressibleEulerEquationsWithGravity2D)
    # Calculate primitive variables and speed of sound
    rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll, phi_ll = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr, phi_rr = u_rr

    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    e_ll = rho_e_ll / rho_ll
    p_ll = (equations.gamma - 1) *
           (rho_e_ll - 1 / 2 * rho_ll * (v1_ll^2 + v2_ll^2) - rho_ll * phi_ll)
    c_ll = sqrt(equations.gamma * p_ll / rho_ll)

    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    e_rr = rho_e_rr / rho_rr
    p_rr = (equations.gamma - 1) *
           (rho_e_rr - 1 / 2 * rho_rr * (v1_rr^2 + v2_rr^2) - rho_rr * phi_rr)
    c_rr = sqrt(equations.gamma * p_rr / rho_rr)

    # Obtain left and right fluxes
    f_ll = flux(u_ll, orientation, equations)
    f_rr = flux(u_rr, orientation, equations)

    # Compute Roe averages
    sqrt_rho_ll = sqrt(rho_ll)
    sqrt_rho_rr = sqrt(rho_rr)
    sum_sqrt_rho = sqrt_rho_ll + sqrt_rho_rr
    if orientation == 1 # x-direction
        vel_L = v1_ll
        vel_R = v1_rr
        ekin_roe = (sqrt_rho_ll * v2_ll + sqrt_rho_rr * v2_rr)^2
    elseif orientation == 2 # y-direction
        vel_L = v2_ll
        vel_R = v2_rr
        ekin_roe = (sqrt_rho_ll * v1_ll + sqrt_rho_rr * v1_rr)^2
    end
    vel_roe = (sqrt_rho_ll * vel_L + sqrt_rho_rr * vel_R) / sum_sqrt_rho
    ekin_roe = 0.5 * (vel_roe^2 + ekin_roe / sum_sqrt_rho^2)
    H_ll = (rho_e_ll + p_ll) / rho_ll
    H_rr = (rho_e_rr + p_rr) / rho_rr
    H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) / sum_sqrt_rho
    c_roe = sqrt((equations.gamma - 1) * (H_roe - ekin_roe))
    Ssl = min(vel_L - c_ll, vel_roe - c_roe)
    Ssr = max(vel_R + c_rr, vel_roe + c_roe)
    sMu_L = Ssl - vel_L
    sMu_R = Ssr - vel_R

    if Ssl >= 0.0
        f1 = f_ll[1]
        f2 = f_ll[2]
        f3 = f_ll[3]
        f4 = f_ll[4]
    elseif Ssr <= 0.0
        f1 = f_rr[1]
        f2 = f_rr[2]
        f3 = f_rr[3]
        f4 = f_rr[4]
    else
        SStar = (p_rr - p_ll + rho_ll * vel_L * sMu_L - rho_rr * vel_R * sMu_R) /
                (rho_ll * sMu_L - rho_rr * sMu_R)
        if Ssl <= 0.0 <= SStar
            densStar = rho_ll * sMu_L / (Ssl - SStar)
            enerStar = e_ll + (SStar - vel_L) * (SStar + p_ll / (rho_ll * sMu_L))
            UStar1 = densStar
            UStar4 = densStar * enerStar
            if orientation == 1 # x-direction
                UStar2 = densStar * SStar
                UStar3 = densStar * v2_ll
            elseif orientation == 2 # y-direction
                UStar2 = densStar * v1_ll
                UStar3 = densStar * SStar
            end
            f1 = f_ll[1] + Ssl * (UStar1 - rho_ll)
            f2 = f_ll[2] + Ssl * (UStar2 - rho_v1_ll)
            f3 = f_ll[3] + Ssl * (UStar3 - rho_v2_ll)
            f4 = f_ll[4] + Ssl * (UStar4 - rho_e_ll)
        else
            densStar = rho_rr * sMu_R / (Ssr - SStar)
            enerStar = e_rr + (SStar - vel_R) * (SStar + p_rr / (rho_rr * sMu_R))
            UStar1 = densStar
            UStar4 = densStar * enerStar
            if orientation == 1 # x-direction
                UStar2 = densStar * SStar
                UStar3 = densStar * v2_rr
            elseif orientation == 2 # y-direction
                UStar2 = densStar * v1_rr
                UStar3 = densStar * SStar
            end
            f1 = f_rr[1] + Ssr * (UStar1 - rho_rr)
            f2 = f_rr[2] + Ssr * (UStar2 - rho_v1_rr)
            f3 = f_rr[3] + Ssr * (UStar3 - rho_v2_rr)
            f4 = f_rr[4] + Ssr * (UStar4 - rho_e_rr)
        end
    end
    return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end

"""
    flux_hlle(u_ll, u_rr, orientation, equations::CompressibleEulerEquationsWithGravity2D)

Computes the HLLE (Harten-Lax-van Leer-Einfeldt) flux for the compressible Euler equations.
Special estimates of the signal velocites and linearization of the Riemann problem developed
by Einfeldt to ensure that the internal energy and density remain positive during the computation
of the numerical flux.

- Bernd Einfeldt (1988)
    On Godunov-type methods for gas dynamics.
    [DOI: 10.1137/0725021](https://doi.org/10.1137/0725021)
- Bernd Einfeldt, Claus-Dieter Munz, Philip L. Roe and Björn Sjögreen (1991)
    On Godunov-type methods near low densities.
    [DOI: 10.1016/0021-9991(91)90211-3](https://doi.org/10.1016/0021-9991(91)90211-3)
"""
function flux_hlle(u_ll, u_rr, orientation::Integer,
                   equations::CompressibleEulerEquationsWithGravity2D)
    # Calculate primitive variables, enthalpy and speed of sound
    rho_ll, v1_ll, v2_ll, p_ll, _ = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr, _ = cons2prim(u_rr, equations)

    # `u_ll[4]` is total energy `rho_e_ll` on the left
    H_ll = (u_ll[4] + p_ll) / rho_ll
    c_ll = sqrt(equations.gamma * p_ll / rho_ll)

    # `u_rr[4]` is total energy `rho_e_rr` on the right
    H_rr = (u_rr[4] + p_rr) / rho_rr
    c_rr = sqrt(equations.gamma * p_rr / rho_rr)

    # Compute Roe averages
    sqrt_rho_ll = sqrt(rho_ll)
    sqrt_rho_rr = sqrt(rho_rr)
    inv_sum_sqrt_rho = inv(sqrt_rho_ll + sqrt_rho_rr)

    v1_roe = (sqrt_rho_ll * v1_ll + sqrt_rho_rr * v1_rr) * inv_sum_sqrt_rho
    v2_roe = (sqrt_rho_ll * v2_ll + sqrt_rho_rr * v2_rr) * inv_sum_sqrt_rho
    v_roe_mag = v1_roe^2 + v2_roe^2

    H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) * inv_sum_sqrt_rho
    c_roe = sqrt((equations.gamma - 1) * (H_roe - 0.5 * v_roe_mag))

    # Compute convenience constant for positivity preservation, see
    # https://doi.org/10.1016/0021-9991(91)90211-3
    beta = sqrt(0.5 * (equations.gamma - 1) / equations.gamma)

    # Estimate the edges of the Riemann fan (with positivity conservation)
    if orientation == 1 # x-direction
        SsL = min(v1_roe - c_roe, v1_ll - beta * c_ll, zero(v1_roe))
        SsR = max(v1_roe + c_roe, v1_rr + beta * c_rr, zero(v1_roe))
    elseif orientation == 2 # y-direction
        SsL = min(v2_roe - c_roe, v2_ll - beta * c_ll, zero(v2_roe))
        SsR = max(v2_roe + c_roe, v2_rr + beta * c_rr, zero(v2_roe))
    end

    if SsL >= 0.0 && SsR > 0.0
        # Positive supersonic speed
        f_ll = flux(u_ll, orientation, equations)

        f1 = f_ll[1]
        f2 = f_ll[2]
        f3 = f_ll[3]
        f4 = f_ll[4]
    elseif SsR <= 0.0 && SsL < 0.0
        # Negative supersonic speed
        f_rr = flux(u_rr, orientation, equations)

        f1 = f_rr[1]
        f2 = f_rr[2]
        f3 = f_rr[3]
        f4 = f_rr[4]
    else
        # Subsonic case
        # Compute left and right fluxes
        f_ll = flux(u_ll, orientation, equations)
        f_rr = flux(u_rr, orientation, equations)

        f1 = (SsR * f_ll[1] - SsL * f_rr[1] + SsL * SsR * (u_rr[1] - u_ll[1])) /
             (SsR - SsL)
        f2 = (SsR * f_ll[2] - SsL * f_rr[2] + SsL * SsR * (u_rr[2] - u_ll[2])) /
             (SsR - SsL)
        f3 = (SsR * f_ll[3] - SsL * f_rr[3] + SsL * SsR * (u_rr[3] - u_ll[3])) /
             (SsR - SsL)
        f4 = (SsR * f_ll[4] - SsL * f_rr[4] + SsL * SsR * (u_rr[4] - u_ll[4])) /
             (SsR - SsL)
    end

    return SVector(f1, f2, f3, f4, zero(eltype(u_ll)))
end

@inline function max_abs_speeds(u,
                                equations::CompressibleEulerEquationsWithGravity2D)
    rho, v1, v2, p, _ = cons2prim(u, equations)
    c = sqrt(equations.gamma * p / rho)

    return abs(v1) + c, abs(v2) + c
end

# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerEquationsWithGravity2D)
    rho, rho_v1, rho_v2, rho_e, phi = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2) - rho * phi)

    return SVector(rho, v1, v2, p, phi)
end

# Convert conservative variables to entropy (see, e.g., Waruszewski et al. (2022))
@inline function cons2entropy(u,
                              equations::CompressibleEulerEquationsWithGravity2D)
    rho, rho_v1, rho_v2, rho_e, phi = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_square = v1^2 + v2^2
    p = (equations.gamma - 1) * (rho_e - 0.5 * rho * v_square - rho * phi)
    s = log(p) - equations.gamma * log(rho)
    rho_p = rho / p

    w1 = (equations.gamma - s) * equations.inv_gamma_minus_one -
         rho_p * (0.5 * v_square - phi)
    w2 = rho_p * v1
    w3 = rho_p * v2
    w4 = -rho_p

    return SVector(w1, w2, w3, w4, phi)
end

@inline function entropy2cons(w,
                              equations::CompressibleEulerEquationsWithGravity2D)
    # See Waruszewski et al. (2022)
    @unpack gamma = equations

    # convert to entropy `-rho * s`
    # instead of `-rho * s / (gamma - 1)`
    V1, V2, V3, V5, _ = w .* (gamma - 1)
    phi = w[5]

    # s = specific entropy
    s = gamma - V1 + (V2^2 + V3^2) / (2 * V5) - V5 * phi

    rho_iota = ((gamma - 1) / (-V5)^gamma)^(equations.inv_gamma_minus_one) *
               exp(-s * equations.inv_gamma_minus_one)

    rho = -rho_iota * V5
    rho_v1 = rho_iota * V2
    rho_v2 = rho_iota * V3
    rho_e = rho_iota * (1 - (V2^2 + V3^2) / (2 * V5)) + rho * phi
    return SVector(rho, rho_v1, rho_v2, rho_e, phi)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim,
                           equations::CompressibleEulerEquationsWithGravity2D)
    rho, v1, v2, p, phi = prim
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_e = p * equations.inv_gamma_minus_one + 0.5 * (rho_v1 * v1 + rho_v2 * v2) +
            rho * phi
    return SVector(rho, rho_v1, rho_v2, rho_e, phi)
end

@inline function density(u, equations::CompressibleEulerEquationsWithGravity2D)
    rho = u[1]
    return rho
end

@inline function pressure(u, equations::CompressibleEulerEquationsWithGravity2D)
    rho, rho_v1, rho_v2, rho_e, phi = u
    p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho - rho * phi)
    return p
end

@inline function density_pressure(u,
                                  equations::CompressibleEulerEquationsWithGravity2D)
    rho, rho_v1, rho_v2, rho_e, phi = u
    rho_times_p = (equations.gamma - 1) *
                  (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2) - rho^2 * phi)
    return rho_times_p
end

# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons,
                                       equations::CompressibleEulerEquationsWithGravity2D)
    # Pressure
    p = (equations.gamma - 1) *
        (cons[4] - 1 / 2 * (cons[2]^2 + cons[3]^2) / cons[1] - cons[1] * cons[5])

    # Thermodynamic entropy
    s = log(p) - equations.gamma * log(cons[1])

    return s
end

# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons,
                              equations::CompressibleEulerEquationsWithGravity2D)
    # Mathematical entropy
    S = -entropy_thermodynamic(cons, equations) * cons[1] *
        equations.inv_gamma_minus_one

    return S
end

# Default entropy is the mathematical entropy
@inline entropy(cons, equations::CompressibleEulerEquationsWithGravity2D) = entropy_math(cons,
                                                                                         equations)

# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::CompressibleEulerEquationsWithGravity2D) = cons[4]

# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u,
                                equations::CompressibleEulerEquationsWithGravity2D)
    rho, rho_v1, rho_v2, rho_e, _ = u
    return (rho_v1^2 + rho_v2^2) / (2 * rho)
end

# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons,
                                 equations::CompressibleEulerEquationsWithGravity2D)
    return energy_total(cons, equations) - energy_kinetic(cons, equations) -
           cons[1] * cons[5]
end

@inline function velocity(u, equations::CompressibleEulerEquationsWithGravity2D)
    rho = u[1]
    v1 = u[2] / rho
    v2 = u[3] / rho
    return SVector(v1, v2)
end

# Specialized `DissipationLocalLaxFriedrichs` to avoid spurious dissipation in the
# gravitational potential
@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr,
                                                              orientation_or_normal_direction,
                                                              equations::CompressibleEulerEquationsWithGravity2D)
    λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction,
                                  equations)
    diss = -0.5 * λ * (u_rr - u_ll)
    return SVector(diss[1], diss[2], diss[3], diss[4], zero(eltype(u_ll)))
end
end # @muladd
