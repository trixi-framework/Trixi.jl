# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent
@doc raw"""
    CompressibleEulerEquationsFullPerturbationGravity2D(gamma)

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
struct CompressibleEulerEquationsFullPerturbationGravity2D{RealT <: Real} <:
       AbstractCompressibleEulerEquations{2, 4}
    equations_total::CompressibleEulerEquations2D{RealT}

    function CompressibleEulerEquationsFullPerturbationGravity2D(gamma)
        γ, inv_gamma_minus_one = promote(gamma, inv(gamma - 1))
        new{typeof(γ)}(CompressibleEulerEquations2D(gamma))
    end
end

have_nonconservative_terms(::CompressibleEulerEquationsFullPerturbationGravity2D) = True()
have_aux_node_vars(::CompressibleEulerEquationsFullPerturbationGravity2D) = True()
n_aux_node_vars(::CompressibleEulerEquationsFullPerturbationGravity2D) = 5

function cons2aux(u, aux, equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    return SVector(aux[1], aux[2], aux[3], aux[4], aux[5])
end

varnames(::typeof(cons2aux), ::CompressibleEulerEquationsFullPerturbationGravity2D) = ("rho_steady",
                                                                                   "rho_v1_steady",
                                                                                   "rho_v2_steady",
                                                                                   "rho_e_steady",
                                                                                   "geopotential")

# Add steady state to current perturbations (in conserved variables)
@inline function cons2cons_total(u, aux,
                                 equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    return u + SVector(aux[1], aux[2], aux[3], aux[4])
end

# will fail when steady rho ~ 0
# only works when geopotential is included in u[4]
@inline function cons2prim_geopot(u, aux,
                                  equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    rho, rho_v1, rho_v2, rho_e = u
    phi = aux[5]
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.equations_total.gamma - 1) *
        (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2) - rho * phi)
    return SVector(rho, v1, v2, p)
end

# Convert total conservative variables to total primitive variables
@inline function cons2prim_total(u, aux,
                                 equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    u_total = cons2cons_total(u, aux, equations)
    return cons2prim_geopot(u_total, aux, equations)
end

varnames(::typeof(cons2prim_total), ::CompressibleEulerEquationsFullPerturbationGravity2D) = ("rho_total",
                                                                                          "v1_total",
                                                                                          "v2_total",
                                                                                          "p_total")

# Convert perturbation in conservative variables to perturbation in primitive variables
# cons2prim applied to perturbations might fail when rho ~ 0
# this will likewise fail when steady rho ~ 0
@inline function cons2prim_pert(u, aux,
                                equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    u_prim_total = cons2prim_total(u, aux, equations)
    u_prim_steady = cons2prim_geopot(aux, aux, equations)
    return u_prim_total - u_prim_steady
end

varnames(::typeof(cons2prim_pert), ::CompressibleEulerEquationsFullPerturbationGravity2D) = ("rho_pert",
                                                                                         "v1_pert",
                                                                                         "v2_pert",
                                                                                         "p_pert")

"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                    equations::CompressibleEulerEquationsFullPerturbationGravity2D)

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
@inline function boundary_condition_slip_wall(u_inner, aux_inner,
                                              normal_direction::AbstractVector,
                                              x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    @unpack gamma, inv_gamma_minus_one = equations.equations_total

    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    # compute total conserved quantities
    u_total = cons2cons_total(u_inner, aux_inner, equations)

    # compute the boundary conditions for total conserved quantities
    p_star_total = calc_pstar(u_total, aux_inner, normal, equations, gamma, inv_gamma_minus_one)
    
    # compute boundary conditions for steady conserved quantities
    p_star_steady = calc_pstar(aux_inner, aux_inner, normal, equations, gamma, inv_gamma_minus_one)
    
    return (SVector(0,
                    (p_star_total - p_star_steady) * normal_direction[1],
                    (p_star_total - p_star_steady) * normal_direction[2],
                    0),
            SVector(0, 0, 0, 0))
end

@inline function calc_pstar(u, aux, normal, equations, gamma, inv_gamma_minus_one)
    # rotate the internal solution state
    u_rot = rotate_to_x(u, normal, equations)

    # compute the primitive variables
    rho_local, v_normal, v_tangent, p_local = cons2prim_geopot(u_rot, aux, equations)

    # Get the solution of the pressure Riemann problem
    # See Section 6.3.3 of
    # Eleuterio F. Toro (2009)
    # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
    # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
    if v_normal <= 0.0
        sound_speed = sqrt(gamma * p_local / rho_local) # local sound speed
        p_star = p_local *
                 (1 + 0.5 * (gamma - 1) * v_normal / sound_speed)^(2 *
                                                                             gamma *
                                                                             inv_gamma_minus_one)
    else # v_normal > 0.0
        A = 2 / ((gamma + 1) * rho_local)
        B = p_local * (gamma - 1) / (gamma + 1)
        p_star = p_local +
                 0.5 * v_normal / A *
                 (v_normal + sqrt(v_normal^2 + 4 * A * (p_local + B)))
    end
    return p_star
end

"""
    boundary_condition_slip_wall(u_inner, aux_inner, orientation, direction, x, t,
                                    surface_flux_function, equations::CompressibleEulerEquationsFullPerturbationGravity2D)

Should be used together with [`TreeMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, aux_inner, orientation,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    # get the appropriate normal vector from the orientation
    if orientation == 1
        normal_direction = SVector(1, 0)
    else # orientation == 2
        normal_direction = SVector(0, 1)
    end

    # compute and return the flux using `boundary_condition_slip_wall` routine above
    return boundary_condition_slip_wall(u_inner, aux_inner, normal_direction, direction,
                                        x, t, surface_flux_function, equations)
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, direction, x, t,
                                    surface_flux_function, equations::CompressibleEulerEquationsFullPerturbationGravity2D)

Should be used together with [`StructuredMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, aux_inner,
                                              normal_direction::AbstractVector,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    # flip sign of normal to make it outward pointing, then flip the sign of the normal flux back
    # to be inward pointing on the -x and -y sides due to the orientation convention used by StructuredMesh
    if isodd(direction)
        fluxes = boundary_condition_slip_wall(u_inner, aux_inner, -normal_direction,
                                              x, t, surface_flux_function, equations)
        boundary_flux = (-fluxes[1], -fluxes[2])
    else
        boundary_flux = boundary_condition_slip_wall(u_inner, aux_inner,
                                                     normal_direction,
                                                     x, t, surface_flux_function,
                                                     equations)
    end

    return boundary_flux
end

# Calculate 2D flux for a single point
@inline function flux(u, aux, orientation::Integer,
                      equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    u_total = cons2cons_total(u, aux, equations)
    flux_total = calc_flux(u_total, aux, orientation, equations)
    flux_steady = calc_flux(aux, aux, orientation, equations)
    return flux_total - flux_steady
end

# Calculate 2D flux for a single point
@inline function calc_flux(u, aux, orientation::Integer,
                      equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    _, rho_v1, rho_v2, rho_e = u
    _, v1, v2, p = cons2prim_geopot(u, aux, equations)
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
    return SVector(f1, f2, f3, f4)
end

"""
    flux_kennedy_gruber(u_ll, u_rr, orientation_or_normal_direction,
                        equations::CompressibleEulerEquationsFullPerturbationGravity2D)

Kinetic energy preserving two-point flux by
- Kennedy and Gruber (2008)
    Reduced aliasing formulations of the convective terms within the
    Navier-Stokes equations for a compressible fluid
    [DOI: 10.1016/j.jcp.2007.09.020](https://doi.org/10.1016/j.jcp.2007.09.020)
"""
@inline function flux_kennedy_gruber(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                                     equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    # compute total conserved quantities
    u_ll_total = cons2cons_total(u_ll, aux_ll, equations)
    u_rr_total = cons2cons_total(u_rr, aux_rr, equations)

    flux_total = calc_flux_kennedy_gruber(u_ll_total, u_rr_total, aux_ll, aux_rr,
                                          orientation, equations)
    flux_steady = calc_flux_kennedy_gruber(aux_ll, aux_rr, aux_ll, aux_rr, orientation,
                                           equations)
    return flux_total - flux_steady
end

@inline function calc_flux_kennedy_gruber(u_ll, u_rr, aux_ll, aux_rr, orientation,
                                          equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    # Unpack left and right state
    rho_e_ll = u_ll[4]
    rho_e_rr = u_rr[4]
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim_geopot(u_ll, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim_geopot(u_rr, aux_rr, equations)

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

    return SVector(f1, f2, f3, f4)
end

function flux_nonconservative_waruszewski(u_ll, u_rr, aux_ll, aux_rr,
                                          orientation::Integer,
                                          equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    rho_ll_total = u_ll[1] + aux_ll[1]
    rho_rr_total = u_rr[1] + aux_rr[1]
    # phi is steady
    phi_ll = aux_ll[5]
    phi_rr = aux_rr[5]

    # We omit the 0.5 in the density average since jl always multiplies the non-conservative flux with 0.5
    noncons = (ln_mean(rho_ll_total, rho_rr_total) - ln_mean(aux_ll[1], aux_rr[1])) *
              (phi_rr - phi_ll)
    #noncons = 0.5 * (rho_ll + rho_rr) * (phi_rr - phi_ll)

    f0 = zero(eltype(u_ll))
    if orientation == 1
        return SVector(f0, noncons, f0, f0)
    else #if orientation == 2
        return SVector(f0, f0, noncons, f0)
    end
end

"""
    FluxLMARS(c)(u_ll, u_rr, orientation_or_normal_direction,
                    equations::CompressibleEulerEquationsFullPerturbationGravity2D)

Low Mach number approximate Riemann solver (LMARS) for atmospheric flows using
an estimate `c` of the speed of sound.

References:
- Xi Chen et al. (2013)
    A Control-Volume Model of the Compressible Euler Equations with a Vertical
    Lagrangian Coordinate
    [DOI: 10.1175/MWR-D-12-00129.1](https://doi.org/10.1175/mwr-d-12-00129.1)
"""
# The struct is already defined in CompressibleEulerEquations2D

@inline function (flux_lmars::FluxLMARS)(u_ll, u_rr, aux_ll, aux_rr,
                                         orientation::Integer,
                                         equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    c = flux_lmars.speed_of_sound

    # compute total conserved quantities
    u_ll_total = cons2cons_total(u_ll, aux_ll, equations)
    u_rr_total = cons2cons_total(u_rr, aux_rr, equations)

    flux_total = calc_flux_lmars(u_ll_total, u_rr_total, aux_ll, aux_rr, c,
                                 orientation, equations)
    flux_steady = calc_flux_lmars(aux_ll, aux_rr, aux_ll, aux_rr, c, orientation, equations)
    return flux_total - flux_steady
end

@inline function calc_flux_lmars(u_ll, u_rr, aux_ll, aux_rr, c, orientation::Integer,
                                 equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim_geopot(u_ll, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim_geopot(u_rr, aux_rr, equations)

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
        f1, f2, f3, f4 = v * u_ll
        f4 = f4 + p_ll * v
    else
        f1, f2, f3, f4 = v * u_rr
        f4 = f4 + p_rr * v
    end

    if orientation == 1
        f2 = f2 + p
    else # orientation == 2
        f3 = f3 + p
    end

    return SVector(f1, f2, f3, f4)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                                     equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim_total(u_ll, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim_total(u_rr, aux_rr, equations)

    # Get the velocity value in the appropriate direction
    if orientation == 1
        v_ll = v1_ll
        v_rr = v1_rr
    else # orientation == 2
        v_ll = v2_ll
        v_rr = v2_rr
    end
    # Calculate sound speeds
    c_ll = sqrt(equations.equations_total.gamma * p_ll / rho_ll)
    c_rr = sqrt(equations.equations_total.gamma * p_rr / rho_rr)

    λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end

# TODO based on total quantities?
@inline function max_abs_speeds(u, aux,
                                equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    rho, v1, v2, p = cons2prim_total(u, aux, equations)
    c = sqrt(equations.equations_total.gamma * p / rho)

    return abs(v1) + c, abs(v2) + c
end

# Convert conservative variables to entropy (see, e.g., Waruszewski et al. (2022))
@inline function cons2entropy(u, aux,
                              equations::CompressibleEulerEquationsFullPerturbationGravity2D)
    @unpack gamma, inv_gamma_minus_one = equations.equations_total
    rho, v1, v2, p = cons2prim_total(u, aux, equations)
    phi = aux[5]

    s = log(p) - gamma * log(rho)
    rho_p = rho / p

    w1 = (gamma - s) * inv_gamma_minus_one -
         rho_p * (0.5 * (v1^2 + v2^2) - phi)
    w2 = rho_p * v1
    w3 = rho_p * v2
    w4 = -rho_p

    # TODO: must phi be included here?
    return SVector(w1, w2, w3, w4)
end
end # @muladd
