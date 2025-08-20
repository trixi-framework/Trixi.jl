# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent
@doc raw"""
    CompressibleEulerEquationsPerturbationGravity3D(gamma)

Like [`CompressibleEulerEquationsWithGravity3D`](@ref), but storing a steady state and the geopotential using auxiliary variabes, and solving for perturbations.
"""
struct CompressibleEulerEquationsPerturbationGravity3D{RealT <: Real} <:
       AbstractCompressibleEulerEquations{3, 5}
    gamma::RealT               # ratio of specific heats
    inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications

    function CompressibleEulerEquationsPerturbationGravity3D(gamma)
        γ, inv_gamma_minus_one = promote(gamma, inv(gamma - 1))
        new{typeof(γ)}(γ, inv_gamma_minus_one)
    end
end

have_nonconservative_terms(::CompressibleEulerEquationsPerturbationGravity3D) = True()
have_aux_node_vars(::CompressibleEulerEquationsPerturbationGravity3D) = True()
n_aux_node_vars(::CompressibleEulerEquationsPerturbationGravity3D) = 6

function cons2aux(u, aux, equations::CompressibleEulerEquationsPerturbationGravity3D)
    return aux
end

varnames(::typeof(cons2aux), ::CompressibleEulerEquationsPerturbationGravity3D) = ("rho_steady",
                                                                                   "rho_v1_steady",
                                                                                   "rho_v2_steady",
                                                                                   "rho_v3_steady",
                                                                                   "rho_e_steady",
                                                                                   "geopotential")

# add steady state to current perturbations (in conserved variables)
@inline function cons2cons_total(u, aux,
                                 equations::CompressibleEulerEquationsPerturbationGravity3D)
    return u + SVector(aux[1], aux[2], aux[3], aux[4], aux[5])
end

# convert conservative to primitive variables
# - will fail when rho (u[1]) ~ 0
# - only works when geopotential is included in u[5]
@inline function cons2prim_geopot(u, aux,
                                  equations::CompressibleEulerEquationsPerturbationGravity3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e = u
    phi = aux[6]
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    p = (equations.gamma - 1) *
        (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3)
         -
         rho * phi)
    return SVector(rho, v1, v2, v3, p)
end

# compute steady pressure from steady state in aux
# - will fail when steady rho ~ 0
# - only works when geopotential is included in steady state
@inline function pressure_steady(aux,
                                 equations::CompressibleEulerEquationsPerturbationGravity3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e, phi = aux
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    p = (equations.gamma - 1) *
        (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3)
         -
         rho * phi)
    return 0
    # return p  TODO
end

# convert primitve to conervative variables
@inline function prim2cons_geopot(prim, phi,
                                  equations::CompressibleEulerEquationsPerturbationGravity3D)
    rho, v1, v2, v3, p = prim
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_v3 = rho * v3
    rho_e = p * equations.inv_gamma_minus_one +
            0.5f0 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3) + rho * phi
    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e)
end

# compute rho times p used in indicators
# TODO: based on total or perturbation?
@inline function density_pressure(u, aux,
                                  equations::CompressibleEulerEquationsPerturbationGravity3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e = u
    #rho, rho_v1, rho_v2, rho_v3, rho_e = cons2cons_total(u, aux, equations)
    phi = aux[6]
    rho_times_p = (equations.gamma - 1) *
                  (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) - rho^2 * phi)
    return rho_times_p
end

# convert total conservative variables to total primitive variables
@inline function cons2prim_total(u, aux,
                                 equations::CompressibleEulerEquationsPerturbationGravity3D)
    u_total = cons2cons_total(u, aux, equations)
    return cons2prim_geopot(u_total, aux, equations)
end

varnames(::typeof(cons2prim_total), ::CompressibleEulerEquationsPerturbationGravity3D) = ("rho_total",
                                                                                          "v1_total",
                                                                                          "v2_total",
                                                                                          "v3_total",
                                                                                          "p_total")

# convert perturbation in conservative variables to perturbations in primitive variables
# - will fail when steady rho ~ 0
@inline function cons2prim_pert(u, aux,
                                equations::CompressibleEulerEquationsPerturbationGravity3D)
    u_prim_total = cons2prim_total(u, aux, equations)
    u_prim_steady = cons2prim_geopot(aux, aux, equations)
    return u_prim_total - u_prim_steady
end

varnames(::typeof(cons2prim_pert), ::CompressibleEulerEquationsPerturbationGravity3D) = ("rho_pert",
                                                                                         "v1_pert",
                                                                                         "v2_pert",
                                                                                         "v3_pert",
                                                                                         "p_pert")

"""
    boundary_condition_slip_wall(u_inner, aux_inner, normal_direction, x, t,
                                 surface_flux_function,
                                 equations::CompressibleEulerEquationsPerturbationGravity3D)

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
                                              x, t, surface_flux_function,
                                              equations::CompressibleEulerEquationsPerturbationGravity3D)
    @unpack gamma, inv_gamma_minus_one = equations

    u_inner_total = cons2cons_total(u_inner, aux_inner, equations)

    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal = normal_direction / norm_

    # Some vector that can't be identical to normal_vector (unless normal_vector == 0)
    tangent1 = SVector(normal_direction[2], normal_direction[3], -normal_direction[1])
    # Orthogonal projection
    tangent1 -= dot(normal, tangent1) * normal
    tangent1 = normalize(tangent1)

    # Third orthogonal vector
    tangent2 = normalize(cross(normal_direction, tangent1))

    # rotate the internal solution state
    u_local = rotate_to_x(u_inner_total, normal, tangent1, tangent2, equations)

    # compute the primitive variables
    rho_local, v_normal, v_tangent1, v_tangent2, p_local = cons2prim_geopot(u_local,
                                                                            aux_inner,
                                                                            equations)

    # Get the solution of the pressure Riemann problem
    # See Section 6.3.3 of
    # Eleuterio F. Toro (2009)
    # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
    # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
    if v_normal <= 0.0
        sound_speed = sqrt(gamma * p_local / rho_local) # local sound speed
        p_star = p_local *
                 (1 + 0.5 * (gamma - 1) * v_normal / sound_speed)^(2 * gamma *
                                                                   inv_gamma_minus_one)
    else # v_normal > 0.0
        A = 2 / ((gamma + 1) * rho_local)
        B = p_local * (gamma - 1) / (gamma + 1)
        p_star = p_local +
                 0.5 * v_normal / A *
                 (v_normal + sqrt(v_normal^2 + 4 * A * (p_local + B)))
    end

    p_steady = pressure_steady(aux_inner, equations)

    # For the slip wall we directly set the flux as the normal velocity is zero
    return (SVector(0,
                    (p_star - p_steady) * normal_direction[1],
                    (p_star - p_steady) * normal_direction[2],
                    (p_star - p_steady) * normal_direction[3],
                    0),
            SVector(0, 0, 0, 0, 0))
end

"""
    boundary_condition_slip_wall(u_inner, aux_inner, orientation, direction, x, t,
                                 surface_flux_function, equations::CompressibleEulerEquationsPerturbationGravity3D)

Should be used together with [`TreeMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, aux_inner, orientation,
                                              direction,
                                              x, t, surface_flux_function,
                                              equations::CompressibleEulerEquationsPerturbationGravity3D)
    # get the appropriate normal vector from the orientation
    if orientation == 1
        normal_direction = SVector(1, 0, 0)
    elseif orientation == 2
        normal_direction = SVector(0, 1, 0)
    else # orientation == 3
        normal_direction = SVector(0, 0, 1)
    end

    # compute and return the flux using `boundary_condition_slip_wall` routine above
    return boundary_condition_slip_wall(u_inner, aux_inner, normal_direction, direction,
                                        x, t, surface_flux_function, equations)
end

"""
    boundary_condition_slip_wall(u_inner, aux_inner, normal_direction, direction, x, t,
                                 surface_flux_function,equations::CompressibleEulerEquationsPerturbationGravity3D)

Should be used together with [`StructuredMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, aux_inner,
                                              normal_direction::AbstractVector,
                                              direction,
                                              x, t, surface_flux_function,
                                              equations::CompressibleEulerEquationsPerturbationGravity3D)
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
                      equations::CompressibleEulerEquationsPerturbationGravity3D)
    u_total = cons2cons_total(u, aux, equations)
    _, rho_v1, rho_v2, rho_v3, rho_e = u_total
    _, v1, v2, v3, p = cons2prim_geopot(u_total, aux, equations)
    p_steady = pressure_steady(aux, equations)
    if orientation == 1
        f1 = rho_v1
        f2 = rho_v1 * v1 + p - p_steady
        f3 = rho_v1 * v2
        f4 = rho_v1 * v3
        f5 = (rho_e + p) * v1
    elseif orientation == 2
        f1 = rho_v2
        f2 = rho_v2 * v1
        f3 = rho_v2 * v2 + p - p_steady
        f4 = rho_v2 * v3
        f5 = (rho_e + p) * v2
    else
        f1 = rho_v3
        f3 = rho_v3 * v1
        f2 = rho_v3 * v2
        f4 = rho_v3 * v3 + p - p_steady
        f5 = (rho_e + p) * v3
    end
    return SVector(f1, f2, f3, f4, f5)
end

# Calculate 2D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, aux, normal_direction::AbstractVector,
                      equations::CompressibleEulerEquationsPerturbationGravity3D)
    u_total = cons2cons_total(u, aux, equations)
    rho, _, _, _, rho_e = u_total
    _, v1, v2, v3, p = cons2prim_geopot(u_total, aux, equations)
    p_steady = pressure_steady(aux, equations)

    v_normal = v1 * normal_direction[1] +
               v2 * normal_direction[2] +
               v3 * normal_direction[3]
    rho_v_normal = rho * v_normal
    f1 = rho_v_normal
    f2 = rho_v_normal * v1 + (p - p_steady) * normal_direction[1]
    f3 = rho_v_normal * v2 + (p - p_steady) * normal_direction[2]
    f4 = rho_v_normal * v3 + (p - p_steady) * normal_direction[3]
    f5 = (rho_e + p) * v_normal
    return SVector(f1, f2, f3, f4, f5)
end

"""
    flux_shima_etal(u_ll, u_rr, aux_ll, aux_rr, orientation_or_normal_direction,
                    equations::CompressibleEulerEquationsPerturbationGravity3D)

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
@inline function flux_shima_etal(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                                 equations::CompressibleEulerEquationsPerturbationGravity3D)
    @unpack inv_gamma_minus_one = equations.equations_total
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim_total(u_ll, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim_total(u_rr, aux_rr, equations)
    p_steady_ll = pressure_steady(aux_ll, equations)
    p_steady_rr = pressure_steady(aux_rr, equations)
    p_steady_avg = 0.5f0 * (p_steady_ll + p_steady_rr)

    # Average each factor of products in flux
    rho_avg = 1 / 2 * (rho_ll + rho_rr)
    v1_avg = 1 / 2 * (v1_ll + v1_rr)
    v2_avg = 1 / 2 * (v2_ll + v2_rr)
    v3_avg = 1 / 2 * (v3_ll + v3_rr)
    p_avg = 1 / 2 * (p_ll + p_rr)
    kin_avg = 1 / 2 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)

    # Calculate fluxes depending on orientation
    if orientation == 1
        pv1_avg = 0.5f0 * (p_ll * v1_rr + p_rr * v1_ll)
        f1 = rho_avg * v1_avg
        f2 = f1 * v1_avg + p_avg - p_steady_avg
        f3 = f1 * v2_avg
        f4 = f1 * v3_avg
        f5 = p_avg * v1_avg * inv_gamma_minus_one + f1 * kin_avg + pv1_avg
    elseif orientation == 2
        pv2_avg = 0.5f0 * (p_ll * v2_rr + p_rr * v2_ll)
        f1 = rho_avg * v2_avg
        f2 = f1 * v1_avg
        f3 = f1 * v2_avg + p_avg - p_steady_avg
        f4 = f1 * v3_avg
        f5 = p_avg * v2_avg * inv_gamma_minus_one + f1 * kin_avg + pv2_avg
    else
        pv3_avg = 0.5f0 * (p_ll * v3_rr + p_rr * v3_ll)
        f1 = rho_avg * v3_avg
        f2 = f1 * v1_avg
        f3 = f1 * v2_avg
        f4 = f1 * v3_avg + p_avg - p_steady_avg
        f5 = p_avg * v3_avg * inv_gamma_minus_one + f1 * kin_avg + pv3_avg
    end

    return SVector(f1, f2, f3, f4, f5)
end

@inline function flux_shima_etal(u_ll, u_rr, aux_ll, aux_rr,
                                 normal_direction::AbstractVector,
                                 equations::CompressibleEulerEquationsPerturbationGravity3D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim_total(u_ll, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim_total(u_rr, aux_rr, equations)
    p_steady_ll = pressure_steady(aux_ll, equations)
    p_steady_rr = pressure_steady(aux_rr, equations)
    p_steady_avg = 0.5f0 * (p_steady_ll + p_steady_rr)
    v_dot_n_ll = v1_ll * normal_direction[1] +
                 v2_ll * normal_direction[2] +
                 v3_ll * normal_direction[3]
    v_dot_n_rr = v1_rr * normal_direction[1] +
                 v2_rr * normal_direction[2] +
                 v3_rr * normal_direction[3]

    # Average each factor of products in flux
    rho_avg = 1 / 2 * (rho_ll + rho_rr)
    v1_avg = 1 / 2 * (v1_ll + v1_rr)
    v2_avg = 1 / 2 * (v2_ll + v2_rr)
    v3_avg = 1 / 2 * (v3_ll + v3_rr)
    v_dot_n_avg = 1 / 2 * (v_dot_n_ll + v_dot_n_rr)
    p_avg = 1 / 2 * (p_ll + p_rr)
    velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_avg * v_dot_n_avg
    f2 = f1 * v1_avg + (p_avg - p_steady_avg) * normal_direction[1]
    f3 = f1 * v2_avg + (p_avg - p_steady_avg) * normal_direction[2]
    f4 = f1 * v3_avg + (p_avg - p_steady_avg) * normal_direction[3]
    f5 = (f1 * velocity_square_avg +
          p_avg * v_dot_n_avg * equations.inv_gamma_minus_one
          + 0.5 * (p_ll * v_dot_n_rr + p_rr * v_dot_n_ll))

    return SVector(f1, f2, f3, f4, f5)
end

"""
    flux_kennedy_gruber(u_ll, u_rr, aux_ll, aux_rr, orientation_or_normal_direction,
                        equations::CompressibleEulerEquationsPerturbationGravity3D)

Kinetic energy preserving two-point flux by
- Kennedy and Gruber (2008)
    Reduced aliasing formulations of the convective terms within the
    Navier-Stokes equations for a compressible fluid
    [DOI: 10.1016/j.jcp.2007.09.020](https://doi.org/10.1016/j.jcp.2007.09.020)
"""
@inline function flux_kennedy_gruber(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                                     equations::CompressibleEulerEquationsPerturbationGravity3D)
    # Unpack left and right state
    u_ll_total = cons2cons_total(u_ll, aux_ll, equations)
    u_rr_total = cons2cons_total(u_rr, aux_rr, equations)
    rho_e_ll = u_ll_total[5]
    rho_e_rr = u_rr_total[5]
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim_geopot(u_ll_total, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim_geopot(u_rr_total, aux_rr, equations)
    p_steady_ll = pressure_steady(aux_ll, equations)
    p_steady_rr = pressure_steady(aux_rr, equations)
    p_steady_avg = 0.5f0 * (p_steady_ll + p_steady_rr)

    # Average each factor of products in flux
    rho_avg = 1 / 2 * (rho_ll + rho_rr)
    v1_avg = 1 / 2 * (v1_ll + v1_rr)
    v2_avg = 1 / 2 * (v2_ll + v2_rr)
    v3_avg = 1 / 2 * (v3_ll + v3_rr)
    p_avg = 1 / 2 * (p_ll + p_rr)
    e_avg = 1 / 2 * (rho_e_ll / rho_ll + rho_e_rr / rho_rr)

    # Calculate fluxes depending on orientation
    if orientation == 1
        f1 = rho_avg * v1_avg
        f2 = rho_avg * v1_avg * v1_avg + p_avg - p_steady_avg
        f3 = rho_avg * v1_avg * v2_avg
        f4 = rho_avg * v1_avg * v3_avg
        f5 = (rho_avg * e_avg + p_avg) * v1_avg
    elseif orientation == 1
        f1 = rho_avg * v2_avg
        f2 = rho_avg * v2_avg * v1_avg
        f3 = rho_avg * v2_avg * v2_avg + p_avg - p_steady_avg
        f4 = rho_avg * v2_avg * v3_avg
        f5 = (rho_avg * e_avg + p_avg) * v2_avg
    else
        f1 = rho_avg * v2_avg
        f2 = rho_avg * v3_avg * v1_avg
        f3 = rho_avg * v3_avg * v2_avg
        f4 = rho_avg * v3_avg * v3_avg + p_avg - p_steady_avg
        f5 = (rho_avg * e_avg + p_avg) * v3_avg
    end

    return SVector(f1, f2, f3, f4, f5)
end

@inline function flux_kennedy_gruber(u_ll, u_rr, aux_ll, aux_rr,
                                     normal_direction::AbstractVector,
                                     equations::CompressibleEulerEquationsPerturbationGravity3D)
    # Unpack left and right state
    u_ll_total = cons2cons_total(u_ll, aux_ll, equations)
    u_rr_total = cons2cons_total(u_rr, aux_rr, equations)
    rho_e_ll = u_ll_total[5]
    rho_e_rr = u_rr_total[5]
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim_geopot(u_ll_total, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim_geopot(u_rr_total, aux_rr, equations)
    p_steady_ll = pressure_steady(aux_ll, equations)
    p_steady_rr = pressure_steady(aux_rr, equations)
    p_steady_avg = 0.5f0 * (p_steady_ll + p_steady_rr)

    # Average each factor of products in flux
    rho_avg = 0.5 * (rho_ll + rho_rr)
    v1_avg = 0.5 * (v1_ll + v1_rr)
    v2_avg = 0.5 * (v2_ll + v2_rr)
    v3_avg = 0.5 * (v3_ll + v3_rr)
    v_dot_n_avg = v1_avg * normal_direction[1] +
                  v2_avg * normal_direction[2] +
                  v3_avg * normal_direction[3]
    p_avg = 0.5 * (p_ll + p_rr)
    e_avg = 0.5 * (rho_e_ll / rho_ll + rho_e_rr / rho_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_avg * v_dot_n_avg
    f2 = f1 * v1_avg + (p_avg - p_steady_avg) * normal_direction[1]
    f3 = f1 * v2_avg + (p_avg - p_steady_avg) * normal_direction[2]
    f4 = f1 * v3_avg + (p_avg - p_steady_avg) * normal_direction[3]
    f5 = f1 * e_avg + p_avg * v_dot_n_avg

    return SVector(f1, f2, f3, f4, f5)
end

"""
    flux_ranocha(u_ll, u_rr, aux_ll, aux_rr, orientation_or_normal_direction,
                 equations::CompressibleEulerEquationsPerturbationGravity3D)

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
@inline function flux_ranocha(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                              equations::CompressibleEulerEquationsPerturbationGravity3D)
    @unpack inv_gamma_minus_one = equations.equations_total
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim_total(u_ll, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim_total(u_rr, aux_rr, equations)
    p_steady_ll = pressure_steady(aux_ll, equations)
    p_steady_rr = pressure_steady(aux_rr, equations)
    p_steady_avg = 0.5f0 * (p_steady_ll + p_steady_rr)

    # Compute the necessary mean values
    rho_mean = ln_mean(rho_ll, rho_rr)
    # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
    # in exact arithmetic since
    #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
    #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
    inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
    v1_avg = 0.5 * (v1_ll + v1_rr)
    v2_avg = 0.5 * (v2_ll + v2_rr)
    v3_avg = 0.5 * (v3_ll + v3_rr)
    p_avg = 0.5 * (p_ll + p_rr)
    velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)

    # Calculate fluxes depending on orientation
    if orientation == 1
        f1 = rho_mean * v1_avg
        f2 = f1 * v1_avg + p_avg - p_steady_avg
        f3 = f1 * v2_avg
        f4 = f1 * v3_avg
        f5 = f1 *
             (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one) +
             0.5 * (p_ll * v1_rr + p_rr * v1_ll)
    elseif orientation == 2
        f1 = rho_mean * v2_avg
        f2 = f1 * v1_avg
        f3 = f1 * v2_avg + p_avg - p_steady_avg
        f4 = f1 * v3_avg
        f5 = f1 *
             (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one) +
             0.5 * (p_ll * v2_rr + p_rr * v2_ll)
    else
        f1 = rho_mean * v3_avg
        f2 = f1 * v1_avg
        f3 = f1 * v2_avg
        f4 = f1 * v3_avg + p_avg - p_steady_avg
        f5 = f1 *
             (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one) +
             0.5 * (p_ll * v3_rr + p_rr * v3_ll)
    end
    return SVector(f1, f2, f3, f4, f5)
end

@inline function flux_ranocha(u_ll, u_rr, aux_ll, aux_rr,
                              normal_direction::AbstractVector,
                              equations::CompressibleEulerEquationsPerturbationGravity3D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim_total(u_ll, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim_total(u_rr, aux_rr, equations)
    p_steady_ll = pressure_steady(aux_ll, equations)
    p_steady_rr = pressure_steady(aux_rr, equations)
    p_steady_avg = 0.5f0 * (p_steady_ll + p_steady_rr)
    v_dot_n_ll = v1_ll * normal_direction[1] +
                 v2_ll * normal_direction[2] +
                 v3_ll * normal_direction[3]
    v_dot_n_rr = v1_rr * normal_direction[1] +
                 v2_rr * normal_direction[2] +
                 v3_rr * normal_direction[3]

    # Compute the necessary mean values
    rho_mean = ln_mean(rho_ll, rho_rr)
    # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
    # in exact arithmetic since
    #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
    #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
    inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
    v1_avg = 0.5 * (v1_ll + v1_rr)
    v2_avg = 0.5 * (v2_ll + v2_rr)
    v3_avg = 0.5 * (v3_ll + v3_rr)
    p_avg = 0.5 * (p_ll + p_rr)
    velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_mean * 0.5 * (v_dot_n_ll + v_dot_n_rr)
    f2 = f1 * v1_avg + (p_avg - p_steady_avg) * normal_direction[1]
    f3 = f1 * v2_avg + (p_avg - p_steady_avg) * normal_direction[2]
    f4 = f1 * v3_avg + (p_avg - p_steady_avg) * normal_direction[3]
    f5 = (f1 * (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one)
          +
          0.5 * (p_ll * v_dot_n_rr + p_rr * v_dot_n_ll))

    return SVector(f1, f2, f3, f4, f5)
end

function flux_nonconservative_waruszewski(u_ll, u_rr, aux_ll, aux_rr,
                                          normal_direction::AbstractVector,
                                          equations::CompressibleEulerEquationsPerturbationGravity3D)
    # u[1] is perturbation in rho
    rho_ll = u_ll[1] + aux_ll[1]
    rho_rr = u_rr[1] + aux_rr[1] # TODO
    phi_ll = aux_ll[6]
    phi_rr = aux_rr[6]

    # We omit the 0.5 in the density average since Trixi.jl always multiplies the non-conservative flux with 0.5
    # noncons = ln_mean(rho_ll, rho_rr) * (phi_rr - phi_ll)
    noncons = 0.5 * (rho_ll + rho_rr) * (phi_rr - phi_ll)

    f0 = zero(eltype(u_ll))
    return SVector(f0,
                   noncons * normal_direction[1],
                   noncons * normal_direction[2],
                   noncons * normal_direction[3],
                   f0)
end

function flux_nonconservative_waruszewski(u_ll, u_rr, aux_ll, aux_rr,
                                          orientation::Integer,
                                          equations::CompressibleEulerEquationsPerturbationGravity3D)
    rho_ll = u_ll[1]
    rho_rr = u_rr[1]
    phi_ll = aux_ll[6]
    phi_rr = aux_rr[6]

    # We omit the 0.5 in the density average since jl always multiplies the non-conservative flux with 0.5
    #noncons = ln_mean(rho_ll, rho_rr) * (phi_rr - phi_ll)
    noncons = 0.5 * (rho_ll + rho_rr) * (phi_rr - phi_ll)

    f0 = zero(eltype(u_ll))
    if orientation == 1
        return SVector(f0, noncons, f0, f0, f0)
    elseif orientation == 2
        return SVector(f0, f0, noncons, f0, f0)
    else #if orientation == 3
        return SVector(f0, f0, f0, noncons, f0)
    end
end

"""
    FluxLMARS(c)(u_ll, u_rr, aux_ll, aux_rr, orientation_or_normal_direction,
                 equations::CompressibleEulerEquationsPerturbationGravity3D)

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
                                         equations::CompressibleEulerEquationsPerturbationGravity3D)
    c = flux_lmars.speed_of_sound

    # Unpack left and right state
    u_ll_total = cons2cons_total(u_ll, aux_ll, equations)
    u_rr_total = cons2cons_total(u_rr, aux_rr, equations)
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim_geopot(u_ll_total, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim_geopot(u_rr_total, aux_rr, equations)
    p_steady_ll = pressure_steady(aux_ll, equations)
    p_steady_rr = pressure_steady(aux_rr, equations)
    p_steady_avg = 0.5f0 * (p_steady_ll + p_steady_rr)

    if orientation == 1
        v_ll = v1_ll
        v_rr = v1_rr
    elseif orientation == 2
        v_ll = v2_ll
        v_rr = v2_rr
    else # orientation == 3
        v_ll = v3_ll
        v_rr = v3_rr
    end

    rho = 0.5f0 * (rho_ll + rho_rr)
    p = 0.5f0 * (p_ll + p_rr) - 0.5f0 * c * rho * (v_rr - v_ll)
    v = 0.5f0 * (v_ll + v_rr) - 1 / (2 * c * rho) * (p_rr - p_ll)

    # We treat the energy term analogous to the potential temperature term in the paper by
    # Chen et al., i.e. we use p_ll and p_rr, and not p
    if v >= 0
        f1, f2, f3, f4, f5 = v * u_ll_total
        f5 = f5 + p_ll * v
    else
        f1, f2, f3, f4, f5 = v * u_rr_total
        f5 = f5 + p_rr * v
    end

    if orientation == 1
        f2 = f2 + p - p_steady_avg
    elseif orientation == 2
        f3 = f3 + p - p_steady_avg
    else # orientation == 2
        f4 = f4 + p - p_steady_avg
    end

    return SVector(f1, f2, f3, f4, f5)
end

@inline function (flux_lmars::FluxLMARS)(u_ll, u_rr, aux_ll, aux_rr,
                                         normal_direction::AbstractVector,
                                         equations::CompressibleEulerEquationsPerturbationGravity3D)
    c = flux_lmars.speed_of_sound

    # Unpack left and right state
    u_ll_total = cons2cons_total(u_ll, aux_ll, equations)
    u_rr_total = cons2cons_total(u_rr, aux_rr, equations)
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim_geopot(u_ll_total, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim_geopot(u_rr_total, aux_rr, equations)
    p_steady_ll = pressure_steady(aux_ll, equations)
    p_steady_rr = pressure_steady(aux_rr, equations)
    p_steady_avg = 0.5f0 * (p_steady_ll + p_steady_rr)

    v_ll = v1_ll * normal_direction[1] +
           v2_ll * normal_direction[2] +
           v3_ll * normal_direction[3]
    v_rr = v1_rr * normal_direction[1] +
           v2_rr * normal_direction[2] +
           v3_rr * normal_direction[3]

    # Note that this is the same as computing v_ll and v_rr with a normalized normal vector
    # and then multiplying v by `norm_` again, but this version is slightly faster.
    norm_ = norm(normal_direction)

    rho = 0.5f0 * (rho_ll + rho_rr)
    p = 0.5f0 * (p_ll + p_rr) - 0.5f0 * c * rho * (v_rr - v_ll) / norm_
    v = 0.5f0 * (v_ll + v_rr) - 1 / (2 * c * rho) * (p_rr - p_ll) * norm_

    # We treat the energy term analogous to the potential temperature term in the paper by
    # Chen et al., i.e. we use p_ll and p_rr, and not p
    if v >= 0
        f1, f2, f3, f4, f5 = u_ll_total * v
        f5 = f5 + p_ll * v
    else
        f1, f2, f3, f4, f5 = u_rr_total * v
        f5 = f5 + p_rr * v
    end

    return SVector(f1,
                   f2 + (p - p_steady_avg) * normal_direction[1],
                   f3 + (p - p_steady_avg) * normal_direction[2],
                   f4 + (p - p_steady_avg) * normal_direction[3],
                   f5)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                                     equations::CompressibleEulerEquationsPerturbationGravity3D)
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim_total(u_ll, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim_total(u_rr, aux_rr, equations)

    # Get the velocity value in the appropriate direction
    if orientation == 1
        v_ll = v1_ll
        v_rr = v1_rr
    elseif orientation == 2
        v_ll = v2_ll
        v_rr = v2_rr
    else # orientation == 3
        v_ll = v3_ll
        v_rr = v3_rr
    end
    # Calculate sound speeds
    c_ll = sqrt(equations.gamma * p_ll / rho_ll)
    c_rr = sqrt(equations.gamma * p_rr / rho_rr)

    λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end

@inline function max_abs_speed_naive(u_ll, u_rr, aux_ll, aux_rr,
                                     normal_direction::AbstractVector,
                                     equations::CompressibleEulerEquationsPerturbationGravity3D)
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim_total(u_ll, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim_total(u_rr, aux_rr, equations)

    # Calculate normal velocities and sound speed
    # left
    v_ll = (v1_ll * normal_direction[1] +
            v2_ll * normal_direction[2] +
            v3_ll * normal_direction[3])
    c_ll = sqrt(equations.gamma * p_ll / rho_ll)
    # right
    v_rr = (v1_rr * normal_direction[1] +
            v2_rr * normal_direction[2] +
            v3_rr * normal_direction[3])
    c_rr = sqrt(equations.gamma * p_rr / rho_rr)

    return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr) * norm(normal_direction)
end

# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                                     equations::CompressibleEulerEquationsPerturbationGravity3D)
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim_total(u_ll, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim_total(u_rr, aux_rr, equations)

    if orientation == 1 # x-direction
        λ_min = v1_ll - sqrt(equations.gamma * p_ll / rho_ll)
        λ_max = v1_rr + sqrt(equations.gamma * p_rr / rho_rr)
    elseif orientation == 2 # y-direction
        λ_min = v2_ll - sqrt(equations.gamma * p_ll / rho_ll)
        λ_max = v2_rr + sqrt(equations.gamma * p_rr / rho_rr)
    else # z-direction
        λ_min = v3_ll - sqrt(equations.gamma * p_ll / rho_ll)
        λ_max = v3_rr + sqrt(equations.gamma * p_rr / rho_rr)
    end

    return λ_min, λ_max
end

@inline function min_max_speed_naive(u_ll, u_rr, aux_ll, aux_rr,
                                     normal_direction::AbstractVector,
                                     equations::CompressibleEulerEquationsPerturbationGravity3D)
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim_total(u_ll, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim_total(u_rr, aux_rr, equations)

    v_normal_ll = v1_ll * normal_direction[1] +
                  v2_ll * normal_direction[2] +
                  v3_ll * normal_direction[3]
    v_normal_rr = v1_rr * normal_direction[1] +
                  v2_rr * normal_direction[2] +
                  v3_rr * normal_direction[3]

    norm_ = norm(normal_direction)
    # The v_normals are already scaled by the norm
    λ_min = v_normal_ll - sqrt(equations.gamma * p_ll / rho_ll) * norm_
    λ_max = v_normal_rr + sqrt(equations.gamma * p_rr / rho_rr) * norm_

    return λ_min, λ_max
end

@inline function max_abs_speeds(u, aux,
                                equations::CompressibleEulerEquationsPerturbationGravity3D)
    rho, v1, v2, v3, p = cons2prim_total(u, aux, equations)
    c = sqrt(equations.gamma * p / rho)

    return abs(v1) + c, abs(v2) + c, abs(v3) + c
end

# Convert conservative variables to entropy (see, e.g., Waruszewski et al. (2022))
# TODO: must phi be included here?
@inline function cons2entropy(u, aux,
                              equations::CompressibleEulerEquationsPerturbationGravity3D)
    rho, v1, v2, v3, p = cons2prim_total(u, aux, equations)
    phi = aux[6]
    v_square = v1^2 + v2^2 + v3^2

    s = log(p) - equations.gamma * log(rho)
    rho_p = rho / p

    w1 = (equations.gamma - s) * equations.inv_gamma_minus_one -
         rho_p * (0.5f0 * v_square - phi)
    w2 = rho_p * v1
    w3 = rho_p * v2
    w4 = rho_p * v3
    w5 = -rho_p

    return SVector(w1, w2, w3, w4, w5)
end
end # @muladd
