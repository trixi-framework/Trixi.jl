# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    CompressibleEulerEquationsPerturbation2D(gamma)

The compressible Euler equations [`CompressibleEulerEquationsPerturbation2D`](@ref), formulated using a steady background state and perturbations in ``\rho`` and ``p``.
"""
struct CompressibleEulerEquationsPerturbation2D{RealT <: Real} <:
       AbstractCompressibleEulerEquations{2, 4}
    gamma::RealT               # ratio of specific heats
    inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications

    function CompressibleEulerEquationsPerturbation2D(gamma)
        γ, inv_gamma_minus_one = promote(gamma, inv(gamma - 1))
        new{typeof(γ)}(γ, inv_gamma_minus_one)
    end
end

have_auxiliary_node_vars(::CompressibleEulerEquationsPerturbation2D) = True()
n_auxiliary_node_vars(::CompressibleEulerEquationsPerturbation2D) = 3

# Convert conservative variables to primitive
#=
@inline function cons2prim(u, equations::CompressibleEulerEquationsPerturbation2D)
    rho, rho_v1, rho_v2, rho_e = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gamma - 1) * (rho_e - 0.5f0 * (rho_v1 * v1 + rho_v2 * v2))

    return SVector(rho, v1, v2, p)
end
=#

@inline function cons2prim_total(u, aux,
                                 equations::CompressibleEulerEquationsPerturbation2D)
    rho, rho_v1, rho_v2, rho_e = cons2cons_total(u, aux, equations)

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gamma - 1) * (rho_e - 0.5f0 * (rho_v1 * rho_v1 + rho_v2 * rho_v2) / rho)

    return SVector(rho, v1, v2, p)
end

@inline function cons2cons_total(u, aux,
                                 equations::CompressibleEulerEquationsPerturbation2D)
    # rho and rho_e are perturbations
    return SVector(u[1] + aux[1], u[2], u[3], u[4] + aux[2])
end

# Convert conservative variables to entropy
# TODO !!!!
@inline function cons2entropy(u, equations::CompressibleEulerEquationsPerturbation2D)
    rho, rho_v1, rho_v2, rho_e = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_square = v1^2 + v2^2
    p = (equations.gamma - 1) * (rho_e - 0.5f0 * rho * v_square)
    s = log(p) - equations.gamma * log(rho)
    rho_p = rho / p

    w1 = (equations.gamma - s) * equations.inv_gamma_minus_one -
         0.5f0 * rho_p * v_square
    w2 = rho_p * v1
    w3 = rho_p * v2
    w4 = -rho_p

    return SVector(w1, w2, w3, w4)
end

@inline function max_abs_speeds(u, aux, equations::CompressibleEulerEquationsPerturbation2D)
    rho, v1, v2, p = cons2prim_total(u, aux, equations)

    # TODO: calculate speed of sound based on total temperature ?
    c = sqrt(equations.gamma * p / rho)

    return abs(v1) + c, abs(v2) + c
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                                     equations::CompressibleEulerEquationsPerturbation2D)
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
    c_ll = sqrt(equations.gamma * p_ll / rho_ll)
    c_rr = sqrt(equations.gamma * p_rr / rho_rr)

    λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end

#=
# Calculate 2D flux for a single point
@inline function flux(u, orientation::Integer,
                      equations::CompressibleEulerEquationsPerturbation2D)
    rho, rho_v1, rho_v2, rho_e = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    # p computed based on total quantities
    # TODO p perturbed?
    p = (equations.gamma - 1) * (rho_e - 0.5f0 * (rho_v1 * v1 + rho_v2 * v2))
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
=#

# Calculate 2D flux for a single point
@inline function flux(u, aux, orientation::Integer,
                      equations::CompressibleEulerEquationsPerturbation2D)
    rho, rho_v1, rho_v2, rho_e = u
    rho_total = rho + aux[1]
    rho_e_total = rho_e + aux[2]
    # p computed based on total quantities
    p_total = (equations.gamma - 1) * (rho_e_total - 0.5f0 * (rho_v1 * rho_v1 + rho_v2 * rho_v2) / rho_total)
    p_pert = (equations.gamma - 1) * (rho_e - 0.5f0 * (rho_v1 * rho_v1 + rho_v2 * rho_v2) / rho_total)
    # TODO override
    #p_pert = p_total - aux[3]
    if orientation == 1
        f1 = rho_v1
        f2 = rho_v1 * rho_v1 / rho_total + p_pert
        f3 = rho_v1 * rho_v2 / rho_total
        f4 = (rho_e_total + p_total) * rho_v1 / rho_total
    else
        f1 = rho_v2
        f2 = rho_v2 * rho_v1 / rho_total
        f3 = rho_v2 * rho_v2 / rho_total +  p_pert
        f4 = (rho_e_total + p_total) * rho_v2 / rho_total
    end
    return SVector(f1, f2, f3, f4)
end

@inline function flux_central(u_ll, u_rr, aux_ll, aux_rr, orientation_or_normal_direction,
                              equations::CompressibleEulerEquationsPerturbation2D)
    # Calculate regular 1D fluxes
    f_ll = flux(u_ll, aux_ll, orientation_or_normal_direction, equations)
    f_rr = flux(u_rr, aux_rr, orientation_or_normal_direction, equations)

    # Average regular fluxes
    return 0.5f0 * (f_ll + f_rr)
end

@inline function (numflux::FluxPlusDissipation)(u_ll, u_rr, aux_ll, aux_rr,
                                                orientation_or_normal_direction,
                                                equations)
    @unpack numerical_flux, dissipation = numflux

    return (numerical_flux(u_ll, u_rr, aux_ll, aux_rr,
                           orientation_or_normal_direction, equations)
            +
            dissipation(u_ll, u_rr, aux_ll, aux_rr,
                        orientation_or_normal_direction, equations))
end

@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr, aux_ll, aux_rr,
                                                              orientation_or_normal_direction,
                                                              equations)
    λ = dissipation.max_abs_speed(u_ll, u_rr, aux_ll, aux_rr,
                                  orientation_or_normal_direction, equations)
    u_total_ll = cons2cons_total(u_ll, aux_ll, equations)
    u_total_rr = cons2cons_total(u_rr, aux_rr, equations)
    return -0.5f0 * λ * (u_total_rr - u_total_ll)
end

"""
    flux_kennedy_gruber(u_ll, u_rr, orientation_or_normal_direction,
                        equations::CompressibleEulerEquations2D)

Kinetic energy preserving two-point flux by
- Kennedy and Gruber (2008)
  Reduced aliasing formulations of the convective terms within the
  Navier-Stokes equations for a compressible fluid
  [DOI: 10.1016/j.jcp.2007.09.020](https://doi.org/10.1016/j.jcp.2007.09.020)
"""
@inline function flux_kennedy_gruber(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                                     equations::CompressibleEulerEquationsPerturbation2D)
    
    # TODO: only perturbed p?

    # Unpack left and right state
    rho_e_ll = last(u_ll) + aux_ll[2]
    rho_e_rr = last(u_rr) + aux_rr[2]
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim_total(u_ll, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim_total(u_rr, aux_rr, equations)

    # Average each factor of products in flux
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    e_avg = 0.5f0 * (rho_e_ll / rho_ll + rho_e_rr / rho_rr)

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

#=
@inline function flux_kennedy_gruber(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::CompressibleEulerEquationsPerturbation2D)
    # Unpack left and right state
    rho_e_ll = last(u_ll)
    rho_e_rr = last(u_rr)
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    # Average each factor of products in flux
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v_dot_n_avg = v1_avg * normal_direction[1] + v2_avg * normal_direction[2]
    p_avg = 0.5f0 * (p_ll + p_rr)
    e_avg = 0.5f0 * (rho_e_ll / rho_ll + rho_e_rr / rho_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_avg * v_dot_n_avg
    f2 = f1 * v1_avg + p_avg * normal_direction[1]
    f3 = f1 * v2_avg + p_avg * normal_direction[2]
    f4 = f1 * e_avg + p_avg * v_dot_n_avg

    return SVector(f1, f2, f3, f4)
end
=#
"""
    FluxLMARS(c)(u_ll, u_rr, orientation_or_normal_direction,
                 equations::CompressibleEulerEquations2D)

Low Mach number approximate Riemann solver (LMARS) for atmospheric flows using
an estimate `c` of the speed of sound.

References:
- Xi Chen et al. (2013)
  A Control-Volume Model of the Compressible Euler Equations with a Vertical
  Lagrangian Coordinate
  [DOI: 10.1175/MWR-D-12-00129.1](https://doi.org/10.1175/mwr-d-12-00129.1)
"""
@inline function (flux_lmars::FluxLMARS)(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                                         equations::CompressibleEulerEquationsPerturbation2D)
    c = flux_lmars.speed_of_sound
# TODO: only perturbed p?
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim_total(u_ll, aux_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim_total(u_rr, aux_rr, equations)

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

"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                 equations::CompressibleEulerEquations2D)

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
                                              equations::CompressibleEulerEquationsPerturbation2D)
                                              norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal = normal_direction / norm_

    # TODO !!! rotate only acts on velocities, anyway
    u_total = cons2cons_total(u_inner, aux_inner, equations)

    # rotate the internal solution state
    u_local = rotate_to_x(u_total, normal, equations)

    # compute the primitive variables
    rho_local, v_normal, v_tangent, p_local = cons2prim(u_local, equations)

    # TODO !!! speed of sound now based on total quantities

    # Get the solution of the pressure Riemann problem
    # See Section 6.3.3 of
    # Eleuterio F. Toro (2009)
    # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
    # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
    if v_normal <= 0
        sound_speed = sqrt(equations.gamma * p_local / rho_local) # local sound speed
        p_star = p_local *
                 (1 + 0.5f0 * (equations.gamma - 1) * v_normal / sound_speed)^(2 *
                                                                               equations.gamma *
                                                                               equations.inv_gamma_minus_one)
    else # v_normal > 0
        A = 2 / ((equations.gamma + 1) * rho_local)
        B = p_local * (equations.gamma - 1) / (equations.gamma + 1)
        p_star = p_local +
                 0.5f0 * v_normal / A *
                 (v_normal + sqrt(v_normal^2 + 4 * A * (p_local + B)))
    end

    # For the slip wall we directly set the flux as the normal velocity is zero
    return SVector(0,
                   p_star * normal[1],
                   p_star * normal[2],
                   0) * norm_
end


@inline function boundary_condition_slip_wall_simple(u_inner, aux_inner,
                                                     orientation, direction,
                                                     x, t,
                                                     surface_flux_function,
                                                     equations::CompressibleEulerEquationsPerturbation2D)
    if orientation == 1
        u2 = -u_inner[2]
        u3 = u_inner[3]
    else
        u2 = u_inner[2]
        u3 = -u_inner[3]
    end

    # create the "external" boundary solution state
    u_boundary = SVector(u_inner[1],
                         u2,
                         u3,
                         u_inner[4])

    # calculate the boundary flux
    if isodd(direction)
        flux = flux_central(u_inner, u_boundary, aux_inner, aux_inner,
                             orientation, equations)
    else
        flux = flux_central(u_inner, u_boundary, aux_inner, aux_inner,
                            orientation, equations)
    end

    return flux
end

"""
    boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function, equations::CompressibleEulerEquations2D)

Should be used together with [`TreeMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, aux_inner, orientation,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquationsPerturbation2D)
    # get the appropriate normal vector from the orientation
    RealT = eltype(u_inner)
    if orientation == 1
        normal_direction = SVector(one(RealT), zero(RealT))
    else # orientation == 2
        normal_direction = SVector(zero(RealT), one(RealT))
    end

    # compute and return the flux using `boundary_condition_slip_wall` routine above
    return boundary_condition_slip_wall(u_inner, aux_inner, normal_direction, direction,
                                        x, t, surface_flux_function, equations)
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, direction, x, t,
                                 surface_flux_function, equations::CompressibleEulerEquations2D)

Should be used together with [`StructuredMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, aux_inner,
                                              normal_direction::AbstractVector,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquationsPerturbation2D)
    # flip sign of normal to make it outward pointing, then flip the sign of the normal flux back
    # to be inward pointing on the -x and -y sides due to the orientation convention used by StructuredMesh
    if isodd(direction)
        boundary_flux = -boundary_condition_slip_wall(u_inner, aux_inner, -normal_direction,
                                                      x, t, surface_flux_function,
                                                      equations)
    else
        boundary_flux = boundary_condition_slip_wall(u_inner, aux_inner, normal_direction,
                                                     x, t, surface_flux_function,
                                                     equations)
    end

    return boundary_flux
end


end # @muladd
