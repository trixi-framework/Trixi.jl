# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    CompressibleEulerEquationsPerturbation2D(gamma)

The compressible Euler equations [`CompressibleEulerEquationsPerturbation2D`](@ref), formulated using a steady background state and perturbations.

Given a state in primitive variables
(``\bar{\rho}``, ``\bar{v}_1``, ``\bar{v}_2``, and ``\bar{p}``),
or conservative variables
(``\bar{\rho}``, ``\bar{\rho} \bar{v}_1``, ``\bar{\rho} \bar{v}_2``, and ``\bar{rho} \bar{e}``),
with ``\bar{rho} \bar{e} = (\gamma -1)^{-1} \bar{p} + 0.5 \bar{rho} |\bar(v)|^2``,
respectively, auxiliary variables are used to store the conservative variables as assumed
steady background state. The equations will then be solved for the perturbations
(``\rho'``, ``(\rho v_1)'``, ``(\rho v_2)'``, and ``(\rho e)'``)
"""
struct CompressibleEulerEquationsPerturbation2D{RealT <: Real} <:
       AbstractCompressibleEulerEquations{2, 4}
    equations_total::CompressibleEulerEquations2D{RealT}

    function CompressibleEulerEquationsPerturbation2D(gamma)
        γ, inv_gamma_minus_one = promote(gamma, inv(gamma - 1))
        new{typeof(γ)}(CompressibleEulerEquations2D(gamma))
    end
end

have_aux_node_vars(::CompressibleEulerEquationsPerturbation2D) = True()
n_aux_node_vars(::CompressibleEulerEquationsPerturbation2D) = 4

function cons2aux(u, aux, equations::CompressibleEulerEquationsPerturbation2D)
    return SVector(aux[1], aux[2], aux[3], aux[4])
end

varnames(::typeof(cons2aux), ::CompressibleEulerEquationsPerturbation2D) =
    ("rho_steady", "rho_v1_steady", "rho_v2_steady", "rho_e_steady")

# Add steady state to current perturbations (in conserved variables)
@inline function cons2cons_total(u, aux,
                                 equations::CompressibleEulerEquationsPerturbation2D)
    return u + SVector(aux[1], aux[2], aux[3], aux[4])
end

# Convert total conservative variables to total primitive variables
@inline function cons2prim_total(u, aux,
                                 equations::CompressibleEulerEquationsPerturbation2D)
    u_cons_total = cons2cons_total(u, aux, equations)
    return cons2prim(u_cons_total, equations.equations_total)
end

varnames(::typeof(cons2prim_total), ::CompressibleEulerEquationsPerturbation2D) =
    ("rho_total", "v1_total", "v2_total", "p_total")

# Convert perturbation in conservative variables to perturbation in primitive variables
# cons2prim applied to perturbations might fail when rho ~ 0
# this will likewise fail when steady rho ~ 0
@inline function cons2prim_pert(u, aux, equations::CompressibleEulerEquationsPerturbation2D)
    u_prim_total = cons2prim_total(u, aux, equations)
    u_prim_steady = cons2prim(aux, equations.equations_total)
    return u_prim_total - u_prim_steady
end

varnames(::typeof(cons2prim_pert), ::CompressibleEulerEquationsPerturbation2D) =
    ("rho_pert", "v1_pert", "v2_pert", "p_pert")

# Convert conservative variables to entropy
@inline function cons2entropy(u, aux, equations::CompressibleEulerEquationsPerturbation2D)
    @unpack gamma, inv_gamma_minus_one = equations.equations_total
    # based on total quantities
    rho, v1, v2, p = cons2prim_total(u, aux, equations)
    v_square = v1^2 + v2^2
    s = log(p) - gamma * log(rho)
    rho_p = rho / p

    w1 = (gamma - s) * inv_gamma_minus_one - 0.5f0 * rho_p * v_square
    w2 = rho_p * v1
    w3 = rho_p * v2
    w4 = -rho_p

    return SVector(w1, w2, w3, w4)
end

@inline function max_abs_speeds(u, aux, equations::CompressibleEulerEquationsPerturbation2D)
    u_cons_total = cons2cons_total(u, aux, equations)
    return max_abs_speeds(u_cons_total, equations.equations_total)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                                     equations::CompressibleEulerEquationsPerturbation2D)
    u_ll_total = cons2cons_total(u_ll, aux_ll, equations)
    u_rr_total = cons2cons_total(u_rr, aux_rr, equations)
    return max_abs_speed_naive(u_ll_total, u_rr_total, orientation,
                               equations.equations_total)
end

# Calculate 2D flux for a single point
# This is special: only p_pert appears
@inline function flux(u, aux, orientation::Integer,
                      equations::CompressibleEulerEquationsPerturbation2D)
    u_total = cons2cons_total(u, aux, equations)
    _flux = flux(u_total, orientation, equations.equations_total)

    # now substract the steady part of p in the momentum equations
    p_steady = pressure(aux, equations.equations_total)
    if orientation == 1
        f2 = p_steady
        f3 = 0
    else # orientation == 2
        f2 = 0
        f3 = p_steady
    end
    return _flux - SVector(0, f2, f3, 0)
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
    u_ll_total = cons2cons_total(u_ll, aux_ll, equations)
    u_rr_total = cons2cons_total(u_rr, aux_rr, equations)
    flux = flux_kennedy_gruber(u_ll_total, u_rr_total, orientation,
                               equations.equations_total)

    # now substract the steady part of p in the momentum equations
    p_steady_avg = 0.5f0 * (pressure(aux_ll, equations.equations_total) +
                            pressure(aux_rr, equations.equations_total))
    if orientation == 1
        f2 = p_steady_avg
        f3 = 0
    else # orientation == 2
        f2 = 0
        f3 = p_steady_avg
    end
    return flux - SVector(0, f2, f3, 0)
end

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
    u_ll_total = cons2cons_total(u_ll, aux_ll, equations)
    u_rr_total = cons2cons_total(u_rr, aux_rr, equations)
    flux = flux_lmars(u_ll_total, u_rr_total, orientation, equations.equations_total)

    # now substract the steady part of p in the momentum equations
    p_steady_avg = 0.5f0 * (pressure(aux_ll, equations.equations_total) +
                            pressure(aux_rr, equations.equations_total))
    if orientation == 1
        f2 = p_steady_avg
        f3 = 0
    else # orientation == 2
        f2 = 0
        f3 = p_steady_avg
    end
    return flux - SVector(0, f2, f3, 0)
end

"""
    flux_shima_etal(u_ll, u_rr, orientation_or_normal_direction,
                    equations::CompressibleEulerEquations2D)

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
                                 equations::CompressibleEulerEquationsPerturbation2D)
    u_ll_total = cons2cons_total(u_ll, aux_ll, equations)
    u_rr_total = cons2cons_total(u_rr, aux_rr, equations)
    flux = flux_shima_etal(u_ll_total, u_rr_total, orientation, equations.equations_total)

    # now substract the steady part of p in the momentum equations
    p_steady_avg = 0.5f0 * (pressure(aux_ll, equations.equations_total) +
                            pressure(aux_rr, equations.equations_total))
    if orientation == 1
        f2 = p_steady_avg
        f3 = 0
    else # orientation == 2
        f2 = 0
        f3 = p_steady_avg
    end
    return flux - SVector(0, f2, f3, 0)
end

"""
    flux_ranocha(u_ll, u_rr, orientation_or_normal_direction,
                 equations::CompressibleEulerEquations2D)

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
                              equations::CompressibleEulerEquationsPerturbation2D)
    u_ll_total = cons2cons_total(u_ll, aux_ll, equations)
    u_rr_total = cons2cons_total(u_rr, aux_rr, equations)
    flux = flux_ranocha(u_ll_total, u_rr_total, orientation, equations.equations_total)

    # now substract the steady part of p in the momentum equations
    p_steady_avg = 0.5f0 * (pressure(aux_ll, equations.equations_total) +
                            pressure(aux_rr, equations.equations_total))
    if orientation == 1
        f2 = p_steady_avg
        f3 = 0
    else # orientation == 2
        f2 = 0
        f3 = p_steady_avg
    end
    return flux - SVector(0, f2, f3, 0)
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
    u_inner_total = cons2cons_total(u_inner, aux_inner, equations)
    boundary_flux = boundary_condition_slip_wall(u_inner_total, normal_direction, x, t,
                                                 surface_flux_function,
                                                 equations.equations_total)
    # This is (0, p_star * normal[1], p_star * normal[2], 0) * norm_
    # now substract the steady part of p
    p_steady = pressure(aux_inner, equations.equations_total)
    return boundary_flux -
           SVector(0, p_steady * normal_direction[1], p_steady * normal_direction[2], 0)
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
