# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    IdealGlmMhdEquations2D(gamma)

The ideal compressible GLM-MHD equations for an ideal gas with ratio of
specific heats `gamma` in two space dimensions.
"""
mutable struct IdealGlmMhdEquations2D{RealT <: Real} <:
               AbstractIdealGlmMhdEquations{2, 9}
    gamma::RealT               # ratio of specific heats
    inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications
    c_h::RealT                 # GLM cleaning speed

    function IdealGlmMhdEquations2D(gamma, c_h)
        γ, inv_gamma_minus_one, c_h = promote(gamma, inv(gamma - 1), c_h)
        new{typeof(γ)}(γ, inv_gamma_minus_one, c_h)
    end
end

function IdealGlmMhdEquations2D(gamma; initial_c_h = convert(typeof(gamma), NaN))
    # Use `promote` to ensure that `gamma` and `initial_c_h` have the same type
    IdealGlmMhdEquations2D(promote(gamma, initial_c_h)...)
end

have_nonconservative_terms(::IdealGlmMhdEquations2D) = True()
n_nonconservative_terms(::IdealGlmMhdEquations2D) = 2

function varnames(::typeof(cons2cons), ::IdealGlmMhdEquations2D)
    ("rho", "rho_v1", "rho_v2", "rho_v3", "rho_e", "B1", "B2", "B3", "psi")
end
function varnames(::typeof(cons2prim), ::IdealGlmMhdEquations2D)
    ("rho", "v1", "v2", "v3", "p", "B1", "B2", "B3", "psi")
end
function default_analysis_integrals(::IdealGlmMhdEquations2D)
    (entropy_timederivative, Val(:l2_divb), Val(:linf_divb))
end

# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::IdealGlmMhdEquations2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equations::IdealGlmMhdEquations2D)
    rho = 1.0
    rho_v1 = 0.1
    rho_v2 = -0.2
    rho_v3 = -0.5
    rho_e = 50.0
    B1 = 3.0
    B2 = -1.2
    B3 = 0.5
    psi = 0.0
    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi)
end

"""
    initial_condition_convergence_test(x, t, equations::IdealGlmMhdEquations2D)

An Alfvén wave as smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::IdealGlmMhdEquations2D)
    # smooth Alfvén wave test from Derigs et al. FLASH (2016)
    # domain must be set to [0, 1/cos(α)] x [0, 1/sin(α)], γ = 5/3
    alpha = 0.25 * pi
    x_perp = x[1] * cos(alpha) + x[2] * sin(alpha)
    B_perp = 0.1 * sin(2.0 * pi * x_perp)
    rho = 1.0
    v1 = -B_perp * sin(alpha)
    v2 = B_perp * cos(alpha)
    v3 = 0.1 * cos(2.0 * pi * x_perp)
    p = 0.1
    B1 = cos(alpha) + v1
    B2 = sin(alpha) + v2
    B3 = v3
    psi = 0.0
    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

"""
    initial_condition_weak_blast_wave(x, t, equations::IdealGlmMhdEquations2D)

A weak blast wave adapted from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t, equations::IdealGlmMhdEquations2D)
    # Adapted MHD version of the weak blast wave from Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
    # Same discontinuity in the velocities but with magnetic fields
    # Set up polar coordinates
    inicenter = (0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)

    # Calculate primitive variables
    rho = r > 0.5 ? 1.0 : 1.1691
    v1 = r > 0.5 ? 0.0 : 0.1882 * cos(phi)
    v2 = r > 0.5 ? 0.0 : 0.1882 * sin(phi)
    p = r > 0.5 ? 1.0 : 1.245

    return prim2cons(SVector(rho, v1, v2, 0.0, p, 1.0, 1.0, 1.0, 0.0), equations)
end

# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::IdealGlmMhdEquations2D)

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::IdealGlmMhdEquations2D)
    rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    kin_en = 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3)
    mag_en = 0.5 * (B1 * B1 + B2 * B2 + B3 * B3)
    p_over_gamma_minus_one = (rho_e - kin_en - mag_en - 0.5 * psi^2)
    p = (equations.gamma - 1) * p_over_gamma_minus_one
    if orientation == 1
        f1 = rho_v1
        f2 = rho_v1 * v1 + p + mag_en - B1^2
        f3 = rho_v1 * v2 - B1 * B2
        f4 = rho_v1 * v3 - B1 * B3
        f5 = (kin_en + equations.gamma * p_over_gamma_minus_one + 2 * mag_en) * v1 -
             B1 * (v1 * B1 + v2 * B2 + v3 * B3) + equations.c_h * psi * B1
        f6 = equations.c_h * psi
        f7 = v1 * B2 - v2 * B1
        f8 = v1 * B3 - v3 * B1
        f9 = equations.c_h * B1
    else #if orientation == 2
        f1 = rho_v2
        f2 = rho_v2 * v1 - B2 * B1
        f3 = rho_v2 * v2 + p + mag_en - B2^2
        f4 = rho_v2 * v3 - B2 * B3
        f5 = (kin_en + equations.gamma * p_over_gamma_minus_one + 2 * mag_en) * v2 -
             B2 * (v1 * B1 + v2 * B2 + v3 * B3) + equations.c_h * psi * B2
        f6 = v2 * B1 - v1 * B2
        f7 = equations.c_h * psi
        f8 = v2 * B3 - v3 * B2
        f9 = equations.c_h * B2
    end

    return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end

# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector,
                      equations::IdealGlmMhdEquations2D)
    rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    kin_en = 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3)
    mag_en = 0.5 * (B1 * B1 + B2 * B2 + B3 * B3)
    p_over_gamma_minus_one = (rho_e - kin_en - mag_en - 0.5 * psi^2)
    p = (equations.gamma - 1) * p_over_gamma_minus_one

    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
    B_normal = B1 * normal_direction[1] + B2 * normal_direction[2]
    rho_v_normal = rho * v_normal

    f1 = rho_v_normal
    f2 = rho_v_normal * v1 - B1 * B_normal + (p + mag_en) * normal_direction[1]
    f3 = rho_v_normal * v2 - B2 * B_normal + (p + mag_en) * normal_direction[2]
    f4 = rho_v_normal * v3 - B3 * B_normal
    f5 = ((kin_en + equations.gamma * p_over_gamma_minus_one + 2 * mag_en) * v_normal
          -
          B_normal * (v1 * B1 + v2 * B2 + v3 * B3) + equations.c_h * psi * B_normal)
    f6 = equations.c_h * psi * normal_direction[1] +
         (v2 * B1 - v1 * B2) * normal_direction[2]
    f7 = equations.c_h * psi * normal_direction[2] +
         (v1 * B2 - v2 * B1) * normal_direction[1]
    f8 = v_normal * B3 - v3 * B_normal
    f9 = equations.c_h * B_normal

    return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end

"""
    flux_nonconservative_powell(u_ll, u_rr, orientation::Integer,
                                equations::IdealGlmMhdEquations2D)
    flux_nonconservative_powell(u_ll, u_rr,
                                normal_direction_ll     ::AbstractVector,
                                normal_direction_average::AbstractVector,
                                equations::IdealGlmMhdEquations2D)

Non-symmetric two-point flux discretizing the nonconservative (source) term of
Powell and the Galilean nonconservative term associated with the GLM multiplier
of the [`IdealGlmMhdEquations2D`](@ref).

On curvilinear meshes, this nonconservative flux depends on both the
contravariant vector (normal direction) at the current node and the averaged
one. This is different from numerical fluxes used to discretize conservative
terms.

## References
- Marvin Bohm, Andrew R.Winters, Gregor J. Gassner, Dominik Derigs,
  Florian Hindenlang, Joachim Saur
  An entropy stable nodal discontinuous Galerkin method for the resistive MHD
  equations. Part I: Theory and numerical verification
  [DOI: 10.1016/j.jcp.2018.06.027](https://doi.org/10.1016/j.jcp.2018.06.027)
"""
@inline function flux_nonconservative_powell(u_ll, u_rr, orientation::Integer,
                                             equations::IdealGlmMhdEquations2D)
    rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    v3_ll = rho_v3_ll / rho_ll
    v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll

    # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
    # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
    if orientation == 1
        f = SVector(0,
                    B1_ll * B1_rr,
                    B2_ll * B1_rr,
                    B3_ll * B1_rr,
                    v_dot_B_ll * B1_rr + v1_ll * psi_ll * psi_rr,
                    v1_ll * B1_rr,
                    v2_ll * B1_rr,
                    v3_ll * B1_rr,
                    v1_ll * psi_rr)
    else # orientation == 2
        f = SVector(0,
                    B1_ll * B2_rr,
                    B2_ll * B2_rr,
                    B3_ll * B2_rr,
                    v_dot_B_ll * B2_rr + v2_ll * psi_ll * psi_rr,
                    v1_ll * B2_rr,
                    v2_ll * B2_rr,
                    v3_ll * B2_rr,
                    v2_ll * psi_rr)
    end

    return f
end

@inline function flux_nonconservative_powell(u_ll, u_rr,
                                             normal_direction_ll::AbstractVector,
                                             normal_direction_average::AbstractVector,
                                             equations::IdealGlmMhdEquations2D)
    rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    v3_ll = rho_v3_ll / rho_ll
    v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll

    # Note that `v_dot_n_ll` uses the `normal_direction_ll` (contravariant vector
    # at the same node location) while `B_dot_n_rr` uses the averaged normal
    # direction. The reason for this is that `v_dot_n_ll` depends only on the left
    # state and multiplies some gradient while `B_dot_n_rr` is used to compute
    # the divergence of B.
    v_dot_n_ll = v1_ll * normal_direction_ll[1] + v2_ll * normal_direction_ll[2]
    B_dot_n_rr = B1_rr * normal_direction_average[1] +
                 B2_rr * normal_direction_average[2]

    # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
    # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
    f = SVector(0,
                B1_ll * B_dot_n_rr,
                B2_ll * B_dot_n_rr,
                B3_ll * B_dot_n_rr,
                v_dot_B_ll * B_dot_n_rr + v_dot_n_ll * psi_ll * psi_rr,
                v1_ll * B_dot_n_rr,
                v2_ll * B_dot_n_rr,
                v3_ll * B_dot_n_rr,
                v_dot_n_ll * psi_rr)

    return f
end

"""
    flux_nonconservative_powell_local_symmetric(u_ll, u_rr,
                                                orientation::Integer,
                                                equations::IdealGlmMhdEquations2D)

Non-symmetric two-point flux discretizing the nonconservative (source) term of
Powell and the Galilean nonconservative term associated with the GLM multiplier
of the [`IdealGlmMhdEquations2D`](@ref).

This implementation uses a non-conservative term that can be written as the product
of local and symmetric parts. It is equivalent to the non-conservative flux of Bohm
et al. (`flux_nonconservative_powell`) for conforming meshes but it yields different
results on non-conforming meshes(!).

The two other flux functions with the same name return either the local 
or symmetric portion of the non-conservative flux based on the type of the 
nonconservative_type argument, employing multiple dispatch. They are used to
compute the subcell fluxes in dg_2d_subcell_limiters.jl.

## References
- Rueda-Ramírez, Gassner (2023). A Flux-Differencing Formula for Split-Form Summation By Parts
  Discretizations of Non-Conservative Systems. https://arxiv.org/pdf/2211.14009.pdf.
"""
@inline function flux_nonconservative_powell_local_symmetric(u_ll, u_rr,
                                                             orientation::Integer,
                                                             equations::IdealGlmMhdEquations2D)
    rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    v3_ll = rho_v3_ll / rho_ll
    v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll

    # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
    # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
    psi_avg = (psi_ll + psi_rr) #* 0.5 # The flux is already multiplied by 0.5 wherever it is used in the code
    if orientation == 1
        B1_avg = (B1_ll + B1_rr) #* 0.5 # The flux is already multiplied by 0.5 wherever it is used in the code
        f = SVector(0,
                    B1_ll * B1_avg,
                    B2_ll * B1_avg,
                    B3_ll * B1_avg,
                    v_dot_B_ll * B1_avg + v1_ll * psi_ll * psi_avg,
                    v1_ll * B1_avg,
                    v2_ll * B1_avg,
                    v3_ll * B1_avg,
                    v1_ll * psi_avg)
    else # orientation == 2
        B2_avg = (B2_ll + B2_rr) #* 0.5 # The flux is already multiplied by 0.5 wherever it is used in the code
        f = SVector(0,
                    B1_ll * B2_avg,
                    B2_ll * B2_avg,
                    B3_ll * B2_avg,
                    v_dot_B_ll * B2_avg + v2_ll * psi_ll * psi_avg,
                    v1_ll * B2_avg,
                    v2_ll * B2_avg,
                    v3_ll * B2_avg,
                    v2_ll * psi_avg)
    end

    return f
end

"""
    flux_nonconservative_powell_local_symmetric(u_ll, orientation::Integer,
                                                equations::IdealGlmMhdEquations2D,
                                                nonconservative_type::NonConservativeLocal,
                                                nonconservative_term::Integer)

Local part of the Powell and GLM non-conservative terms. Needed for the calculation of
the non-conservative staggered "fluxes" for subcell limiting. See, e.g.,
- Rueda-Ramírez, Gassner (2023). A Flux-Differencing Formula for Split-Form Summation By Parts
  Discretizations of Non-Conservative Systems. https://arxiv.org/pdf/2211.14009.pdf.
This function is used to compute the subcell fluxes in dg_2d_subcell_limiters.jl.
"""
@inline function flux_nonconservative_powell_local_symmetric(u_ll, orientation::Integer,
                                                             equations::IdealGlmMhdEquations2D,
                                                             nonconservative_type::NonConservativeLocal,
                                                             nonconservative_term::Integer)
    rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll

    if nonconservative_term == 1
        # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
        v1_ll = rho_v1_ll / rho_ll
        v2_ll = rho_v2_ll / rho_ll
        v3_ll = rho_v3_ll / rho_ll
        v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll
        f = SVector(0,
                    B1_ll,
                    B2_ll,
                    B3_ll,
                    v_dot_B_ll,
                    v1_ll,
                    v2_ll,
                    v3_ll,
                    0)
    else #nonconservative_term ==2
        # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
        if orientation == 1
            v1_ll = rho_v1_ll / rho_ll
            f = SVector(0,
                        0,
                        0,
                        0,
                        v1_ll * psi_ll,
                        0,
                        0,
                        0,
                        v1_ll)
        else #orientation == 2
            v2_ll = rho_v2_ll / rho_ll
            f = SVector(0,
                        0,
                        0,
                        0,
                        v2_ll * psi_ll,
                        0,
                        0,
                        0,
                        v2_ll)
        end
    end
    return f
end

"""
    flux_nonconservative_powell_local_symmetric(u_ll, orientation::Integer,
                                                equations::IdealGlmMhdEquations2D,
                                                nonconservative_type::NonConservativeSymmetric,
                                                nonconservative_term::Integer)

Symmetric part of the Powell and GLM non-conservative terms. Needed for the calculation of
the non-conservative staggered "fluxes" for subcell limiting. See, e.g.,
- Rueda-Ramírez, Gassner (2023). A Flux-Differencing Formula for Split-Form Summation By Parts
  Discretizations of Non-Conservative Systems. https://arxiv.org/pdf/2211.14009.pdf.
This function is used to compute the subcell fluxes in dg_2d_subcell_limiters.jl.
"""
@inline function flux_nonconservative_powell_local_symmetric(u_ll, u_rr,
                                                             orientation::Integer,
                                                             equations::IdealGlmMhdEquations2D,
                                                             nonconservative_type::NonConservativeSymmetric,
                                                             nonconservative_term::Integer)
    rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

    if nonconservative_term == 1
        # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
        if orientation == 1
            B1_avg = (B1_ll + B1_rr)#* 0.5 # The flux is already multiplied by 0.5 wherever it is used in the code
            f = SVector(0,
                        B1_avg,
                        B1_avg,
                        B1_avg,
                        B1_avg,
                        B1_avg,
                        B1_avg,
                        B1_avg,
                        0)
        else # orientation == 2
            B2_avg = (B2_ll + B2_rr)#* 0.5 # The flux is already multiplied by 0.5 wherever it is used in the code
            f = SVector(0,
                        B2_avg,
                        B2_avg,
                        B2_avg,
                        B2_avg,
                        B2_avg,
                        B2_avg,
                        B2_avg,
                        0)
        end
    else #nonconservative_term == 2
        # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
        psi_avg = (psi_ll + psi_rr)#* 0.5 # The flux is already multiplied by 0.5 wherever it is used in the code
        f = SVector(0,
                    0,
                    0,
                    0,
                    psi_avg,
                    0,
                    0,
                    0,
                    psi_avg)
    end

    return f
end

"""
    flux_derigs_etal(u_ll, u_rr, orientation, equations::IdealGlmMhdEquations2D)

Entropy conserving two-point flux by
- Derigs et al. (2018)
  Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field
  divergence diminishing ideal magnetohydrodynamics equations
  [DOI: 10.1016/j.jcp.2018.03.002](https://doi.org/10.1016/j.jcp.2018.03.002)
"""
function flux_derigs_etal(u_ll, u_rr, orientation::Integer,
                          equations::IdealGlmMhdEquations2D)
    # Unpack left and right states to get velocities, pressure, and inverse temperature (called beta)
    rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    v3_ll = rho_v3_ll / rho_ll
    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    v3_rr = rho_v3_rr / rho_rr
    vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
    vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2
    mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
    mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
    p_ll = (equations.gamma - 1) *
           (rho_e_ll - 0.5 * rho_ll * vel_norm_ll - 0.5 * mag_norm_ll - 0.5 * psi_ll^2)
    p_rr = (equations.gamma - 1) *
           (rho_e_rr - 0.5 * rho_rr * vel_norm_rr - 0.5 * mag_norm_rr - 0.5 * psi_rr^2)
    beta_ll = 0.5 * rho_ll / p_ll
    beta_rr = 0.5 * rho_rr / p_rr
    # for convenience store v⋅B
    vel_dot_mag_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll
    vel_dot_mag_rr = v1_rr * B1_rr + v2_rr * B2_rr + v3_rr * B3_rr

    # Compute the necessary mean values needed for either direction
    rho_avg = 0.5 * (rho_ll + rho_rr)
    rho_mean = ln_mean(rho_ll, rho_rr)
    beta_mean = ln_mean(beta_ll, beta_rr)
    beta_avg = 0.5 * (beta_ll + beta_rr)
    v1_avg = 0.5 * (v1_ll + v1_rr)
    v2_avg = 0.5 * (v2_ll + v2_rr)
    v3_avg = 0.5 * (v3_ll + v3_rr)
    p_mean = 0.5 * rho_avg / beta_avg
    B1_avg = 0.5 * (B1_ll + B1_rr)
    B2_avg = 0.5 * (B2_ll + B2_rr)
    B3_avg = 0.5 * (B3_ll + B3_rr)
    psi_avg = 0.5 * (psi_ll + psi_rr)
    vel_norm_avg = 0.5 * (vel_norm_ll + vel_norm_rr)
    mag_norm_avg = 0.5 * (mag_norm_ll + mag_norm_rr)
    vel_dot_mag_avg = 0.5 * (vel_dot_mag_ll + vel_dot_mag_rr)

    # Calculate fluxes depending on orientation with specific direction averages
    if orientation == 1
        f1 = rho_mean * v1_avg
        f2 = f1 * v1_avg + p_mean + 0.5 * mag_norm_avg - B1_avg * B1_avg
        f3 = f1 * v2_avg - B1_avg * B2_avg
        f4 = f1 * v3_avg - B1_avg * B3_avg
        f6 = equations.c_h * psi_avg
        f7 = v1_avg * B2_avg - v2_avg * B1_avg
        f8 = v1_avg * B3_avg - v3_avg * B1_avg
        f9 = equations.c_h * B1_avg
        # total energy flux is complicated and involves the previous eight components
        psi_B1_avg = 0.5 * (B1_ll * psi_ll + B1_rr * psi_rr)
        v1_mag_avg = 0.5 * (v1_ll * mag_norm_ll + v1_rr * mag_norm_rr)
        f5 = (f1 * 0.5 * (1 / (equations.gamma - 1) / beta_mean - vel_norm_avg) +
              f2 * v1_avg + f3 * v2_avg +
              f4 * v3_avg + f6 * B1_avg + f7 * B2_avg + f8 * B3_avg + f9 * psi_avg -
              0.5 * v1_mag_avg +
              B1_avg * vel_dot_mag_avg - equations.c_h * psi_B1_avg)
    else
        f1 = rho_mean * v2_avg
        f2 = f1 * v1_avg - B1_avg * B2_avg
        f3 = f1 * v2_avg + p_mean + 0.5 * mag_norm_avg - B2_avg * B2_avg
        f4 = f1 * v3_avg - B2_avg * B3_avg
        f6 = v2_avg * B1_avg - v1_avg * B2_avg
        f7 = equations.c_h * psi_avg
        f8 = v2_avg * B3_avg - v3_avg * B2_avg
        f9 = equations.c_h * B2_avg
        # total energy flux is complicated and involves the previous eight components
        psi_B2_avg = 0.5 * (B2_ll * psi_ll + B2_rr * psi_rr)
        v2_mag_avg = 0.5 * (v2_ll * mag_norm_ll + v2_rr * mag_norm_rr)
        f5 = (f1 * 0.5 * (1 / (equations.gamma - 1) / beta_mean - vel_norm_avg) +
              f2 * v1_avg + f3 * v2_avg +
              f4 * v3_avg + f6 * B1_avg + f7 * B2_avg + f8 * B3_avg + f9 * psi_avg -
              0.5 * v2_mag_avg +
              B2_avg * vel_dot_mag_avg - equations.c_h * psi_B2_avg)
    end

    return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end

"""
    flux_hindenlang_gassner(u_ll, u_rr, orientation_or_normal_direction,
                            equations::IdealGlmMhdEquations2D)

Entropy conserving and kinetic energy preserving two-point flux of
Hindenlang and Gassner (2019), extending [`flux_ranocha`](@ref) to the MHD equations.

## References
- Florian Hindenlang, Gregor Gassner (2019)
  A new entropy conservative two-point flux for ideal MHD equations derived from
  first principles.
  Presented at HONOM 2019: European workshop on high order numerical methods
  for evolutionary PDEs, theory and applications
- Hendrik Ranocha (2018)
  Generalised Summation-by-Parts Operators and Entropy Stability of Numerical Methods
  for Hyperbolic Balance Laws
  [PhD thesis, TU Braunschweig](https://cuvillier.de/en/shop/publications/7743)
- Hendrik Ranocha (2020)
  Entropy Conserving and Kinetic Energy Preserving Numerical Methods for
  the Euler Equations Using Summation-by-Parts Operators
  [Proceedings of ICOSAHOM 2018](https://doi.org/10.1007/978-3-030-39647-3_42)
"""
@inline function flux_hindenlang_gassner(u_ll, u_rr, orientation::Integer,
                                         equations::IdealGlmMhdEquations2D)
    # Unpack left and right states
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll, B1_ll, B2_ll, B3_ll, psi_ll = cons2prim(u_ll,
                                                                               equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr, B1_rr, B2_rr, B3_rr, psi_rr = cons2prim(u_rr,
                                                                               equations)

    # Compute the necessary mean values needed for either direction
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
    psi_avg = 0.5 * (psi_ll + psi_rr)
    velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)
    magnetic_square_avg = 0.5 * (B1_ll * B1_rr + B2_ll * B2_rr + B3_ll * B3_rr)

    # Calculate fluxes depending on orientation with specific direction averages
    if orientation == 1
        f1 = rho_mean * v1_avg
        f2 = f1 * v1_avg + p_avg + magnetic_square_avg -
             0.5 * (B1_ll * B1_rr + B1_rr * B1_ll)
        f3 = f1 * v2_avg - 0.5 * (B1_ll * B2_rr + B1_rr * B2_ll)
        f4 = f1 * v3_avg - 0.5 * (B1_ll * B3_rr + B1_rr * B3_ll)
        #f5 below
        f6 = equations.c_h * psi_avg
        f7 = 0.5 * (v1_ll * B2_ll - v2_ll * B1_ll + v1_rr * B2_rr - v2_rr * B1_rr)
        f8 = 0.5 * (v1_ll * B3_ll - v3_ll * B1_ll + v1_rr * B3_rr - v3_rr * B1_rr)
        f9 = equations.c_h * 0.5 * (B1_ll + B1_rr)
        # total energy flux is complicated and involves the previous components
        f5 = (f1 *
              (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one)
              +
              0.5 * (+p_ll * v1_rr + p_rr * v1_ll
               + (v1_ll * B2_ll * B2_rr + v1_rr * B2_rr * B2_ll)
               + (v1_ll * B3_ll * B3_rr + v1_rr * B3_rr * B3_ll)
               -
               (v2_ll * B1_ll * B2_rr + v2_rr * B1_rr * B2_ll)
               -
               (v3_ll * B1_ll * B3_rr + v3_rr * B1_rr * B3_ll)
               +
               equations.c_h * (B1_ll * psi_rr + B1_rr * psi_ll)))
    else # orientation == 2
        f1 = rho_mean * v2_avg
        f2 = f1 * v1_avg - 0.5 * (B2_ll * B1_rr + B2_rr * B1_ll)
        f3 = f1 * v2_avg + p_avg + magnetic_square_avg -
             0.5 * (B2_ll * B2_rr + B2_rr * B2_ll)
        f4 = f1 * v3_avg - 0.5 * (B2_ll * B3_rr + B2_rr * B3_ll)
        #f5 below
        f6 = 0.5 * (v2_ll * B1_ll - v1_ll * B2_ll + v2_rr * B1_rr - v1_rr * B2_rr)
        f7 = equations.c_h * psi_avg
        f8 = 0.5 * (v2_ll * B3_ll - v3_ll * B2_ll + v2_rr * B3_rr - v3_rr * B2_rr)
        f9 = equations.c_h * 0.5 * (B2_ll + B2_rr)
        # total energy flux is complicated and involves the previous components
        f5 = (f1 *
              (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one)
              +
              0.5 * (+p_ll * v2_rr + p_rr * v2_ll
               + (v2_ll * B1_ll * B1_rr + v2_rr * B1_rr * B1_ll)
               + (v2_ll * B3_ll * B3_rr + v2_rr * B3_rr * B3_ll)
               -
               (v1_ll * B2_ll * B1_rr + v1_rr * B2_rr * B1_ll)
               -
               (v3_ll * B2_ll * B3_rr + v3_rr * B2_rr * B3_ll)
               +
               equations.c_h * (B2_ll * psi_rr + B2_rr * psi_ll)))
    end

    return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end

@inline function flux_hindenlang_gassner(u_ll, u_rr, normal_direction::AbstractVector,
                                         equations::IdealGlmMhdEquations2D)
    # Unpack left and right states
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll, B1_ll, B2_ll, B3_ll, psi_ll = cons2prim(u_ll,
                                                                               equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr, B1_rr, B2_rr, B3_rr, psi_rr = cons2prim(u_rr,
                                                                               equations)
    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
    B_dot_n_ll = B1_ll * normal_direction[1] + B2_ll * normal_direction[2]
    B_dot_n_rr = B1_rr * normal_direction[1] + B2_rr * normal_direction[2]

    # Compute the necessary mean values needed for either direction
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
    psi_avg = 0.5 * (psi_ll + psi_rr)
    velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)
    magnetic_square_avg = 0.5 * (B1_ll * B1_rr + B2_ll * B2_rr + B3_ll * B3_rr)

    # Calculate fluxes depending on normal_direction
    f1 = rho_mean * 0.5 * (v_dot_n_ll + v_dot_n_rr)
    f2 = (f1 * v1_avg + (p_avg + magnetic_square_avg) * normal_direction[1]
          -
          0.5 * (B_dot_n_ll * B1_rr + B_dot_n_rr * B1_ll))
    f3 = (f1 * v2_avg + (p_avg + magnetic_square_avg) * normal_direction[2]
          -
          0.5 * (B_dot_n_ll * B2_rr + B_dot_n_rr * B2_ll))
    f4 = (f1 * v3_avg
          -
          0.5 * (B_dot_n_ll * B3_rr + B_dot_n_rr * B3_ll))
    #f5 below
    f6 = (equations.c_h * psi_avg * normal_direction[1]
          +
          0.5 * (v_dot_n_ll * B1_ll - v1_ll * B_dot_n_ll +
           v_dot_n_rr * B1_rr - v1_rr * B_dot_n_rr))
    f7 = (equations.c_h * psi_avg * normal_direction[2]
          +
          0.5 * (v_dot_n_ll * B2_ll - v2_ll * B_dot_n_ll +
           v_dot_n_rr * B2_rr - v2_rr * B_dot_n_rr))
    f8 = +0.5 * (v_dot_n_ll * B3_ll - v3_ll * B_dot_n_ll +
          v_dot_n_rr * B3_rr - v3_rr * B_dot_n_rr)
    f9 = equations.c_h * 0.5 * (B_dot_n_ll + B_dot_n_rr)
    # total energy flux is complicated and involves the previous components
    f5 = (f1 * (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one)
          +
          0.5 * (+p_ll * v_dot_n_rr + p_rr * v_dot_n_ll
           + (v_dot_n_ll * B1_ll * B1_rr + v_dot_n_rr * B1_rr * B1_ll)
           + (v_dot_n_ll * B2_ll * B2_rr + v_dot_n_rr * B2_rr * B2_ll)
           + (v_dot_n_ll * B3_ll * B3_rr + v_dot_n_rr * B3_rr * B3_ll)
           -
           (v1_ll * B_dot_n_ll * B1_rr + v1_rr * B_dot_n_rr * B1_ll)
           -
           (v2_ll * B_dot_n_ll * B2_rr + v2_rr * B_dot_n_rr * B2_ll)
           -
           (v3_ll * B_dot_n_ll * B3_rr + v3_rr * B_dot_n_rr * B3_ll)
           +
           equations.c_h * (B_dot_n_ll * psi_rr + B_dot_n_rr * psi_ll)))

    return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::IdealGlmMhdEquations2D)
    rho_ll, rho_v1_ll, rho_v2_ll, _ = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, _ = u_rr

    # Calculate the left/right velocities and fast magnetoacoustic wave speeds
    if orientation == 1
        v_ll = rho_v1_ll / rho_ll
        v_rr = rho_v1_rr / rho_rr
    else # orientation == 2
        v_ll = rho_v2_ll / rho_ll
        v_rr = rho_v2_rr / rho_rr
    end
    cf_ll = calc_fast_wavespeed(u_ll, orientation, equations)
    cf_rr = calc_fast_wavespeed(u_rr, orientation, equations)

    return max(abs(v_ll), abs(v_rr)) + max(cf_ll, cf_rr)
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::IdealGlmMhdEquations2D)
    # return max(v_mag_ll, v_mag_rr) + max(cf_ll, cf_rr)
    rho_ll, rho_v1_ll, rho_v2_ll, _ = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, _ = u_rr

    # Calculate normal velocities and fast magnetoacoustic wave speeds
    # left
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    v_ll = (v1_ll * normal_direction[1]
            +
            v2_ll * normal_direction[2])
    cf_ll = calc_fast_wavespeed(u_ll, normal_direction, equations)
    # right
    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    v_rr = (v1_rr * normal_direction[1]
            +
            v2_rr * normal_direction[2])
    cf_rr = calc_fast_wavespeed(u_rr, normal_direction, equations)

    # wave speeds already scaled by norm(normal_direction) in [`calc_fast_wavespeed`](@ref)
    return max(abs(v_ll), abs(v_rr)) + max(cf_ll, cf_rr)
end

# Calculate estimate for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::IdealGlmMhdEquations2D)
    rho_ll, rho_v1_ll, rho_v2_ll, _ = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, _ = u_rr

    # Calculate primitive velocity variables
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll

    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr

    # Approximate the left-most and right-most eigenvalues in the Riemann fan
    if orientation == 1 # x-direction
        λ_min = v1_ll - calc_fast_wavespeed(u_ll, orientation, equations)
        λ_max = v1_rr + calc_fast_wavespeed(u_rr, orientation, equations)
    else # y-direction
        λ_min = v2_ll - calc_fast_wavespeed(u_ll, orientation, equations)
        λ_max = v2_rr + calc_fast_wavespeed(u_rr, orientation, equations)
    end

    return λ_min, λ_max
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::IdealGlmMhdEquations2D)
    rho_ll, rho_v1_ll, rho_v2_ll, _ = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, _ = u_rr

    # Calculate primitive velocity variables
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll

    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr

    v_normal_ll = (v1_ll * normal_direction[1] + v2_ll * normal_direction[2])
    v_normal_rr = (v1_rr * normal_direction[1] + v2_rr * normal_direction[2])

    c_f_ll = calc_fast_wavespeed(u_ll, normal_direction, equations)
    c_f_rr = calc_fast_wavespeed(u_rr, normal_direction, equations)

    # Estimate the min/max eigenvalues in the normal direction
    λ_min = min(v_normal_ll - c_f_ll, v_normal_rr - c_f_rr)
    λ_max = max(v_normal_rr + c_f_rr, v_normal_rr + c_f_rr)

    return λ_min, λ_max
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::IdealGlmMhdEquations2D)
    rho_ll, rho_v1_ll, rho_v2_ll, _ = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, _ = u_rr

    # Calculate primitive velocity variables
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll

    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr

    # Approximate the left-most and right-most eigenvalues in the Riemann fan
    if orientation == 1 # x-direction
        c_f_ll = calc_fast_wavespeed(u_ll, orientation, equations)
        c_f_rr = calc_fast_wavespeed(u_rr, orientation, equations)

        λ_min = min(v1_ll - c_f_ll, v1_rr - c_f_rr)
        λ_max = max(v1_ll + c_f_ll, v1_rr + c_f_rr)
    else # y-direction
        c_f_ll = calc_fast_wavespeed(u_ll, orientation, equations)
        c_f_rr = calc_fast_wavespeed(u_rr, orientation, equations)

        λ_min = min(v2_ll - c_f_ll, v2_rr - c_f_rr)
        λ_max = max(v2_ll + c_f_ll, v1_rr + c_f_rr)
    end

    return λ_min, λ_max
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::IdealGlmMhdEquations2D)
    rho_ll, rho_v1_ll, rho_v2_ll, _ = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, _ = u_rr

    # Calculate primitive velocity variables
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll

    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr

    v_normal_ll = (v1_ll * normal_direction[1] + v2_ll * normal_direction[2])
    v_normal_rr = (v1_rr * normal_direction[1] + v2_rr * normal_direction[2])

    c_f_ll = calc_fast_wavespeed(u_ll, normal_direction, equations)
    c_f_rr = calc_fast_wavespeed(u_rr, normal_direction, equations)

    # Estimate the min/max eigenvalues in the normal direction
    λ_min = min(v_normal_ll - c_f_ll, v_normal_rr - c_f_rr)
    λ_max = max(v_normal_ll + c_f_ll, v_normal_rr + c_f_rr)

    return λ_min, λ_max
end

"""
    min_max_speed_einfeldt(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdEquations2D)

Calculate minimum and maximum wave speeds for HLL-type fluxes as in
- Li (2005)
  An HLLC Riemann solver for magneto-hydrodynamics
  [DOI: 10.1016/j.jcp.2004.08.020](https://doi.org/10.1016/j.jcp.2004.08.020).
"""
@inline function min_max_speed_einfeldt(u_ll, u_rr, orientation::Integer,
                                        equations::IdealGlmMhdEquations2D)
    rho_ll, rho_v1_ll, rho_v2_ll, _ = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, _ = u_rr

    # Calculate primitive velocity variables
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll

    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr

    # Approximate the left-most and right-most eigenvalues in the Riemann fan
    if orientation == 1 # x-direction
        c_f_ll = calc_fast_wavespeed(u_ll, orientation, equations)
        c_f_rr = calc_fast_wavespeed(u_rr, orientation, equations)
        vel_roe, c_f_roe = calc_fast_wavespeed_roe(u_ll, u_rr, orientation, equations)
        λ_min = min(v1_ll - c_f_ll, vel_roe - c_f_roe)
        λ_max = max(v1_rr + c_f_rr, vel_roe + c_f_roe)
    else # y-direction
        c_f_ll = calc_fast_wavespeed(u_ll, orientation, equations)
        c_f_rr = calc_fast_wavespeed(u_rr, orientation, equations)
        vel_roe, c_f_roe = calc_fast_wavespeed_roe(u_ll, u_rr, orientation, equations)
        λ_min = min(v2_ll - c_f_ll, vel_roe - c_f_roe)
        λ_max = max(v2_rr + c_f_rr, vel_roe + c_f_roe)
    end

    return λ_min, λ_max
end

@inline function min_max_speed_einfeldt(u_ll, u_rr, normal_direction::AbstractVector,
                                        equations::IdealGlmMhdEquations2D)
    rho_ll, rho_v1_ll, rho_v2_ll, _ = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, _ = u_rr

    # Calculate primitive velocity variables
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll

    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr

    v_normal_ll = (v1_ll * normal_direction[1] + v2_ll * normal_direction[2])
    v_normal_rr = (v1_rr * normal_direction[1] + v2_rr * normal_direction[2])

    c_f_ll = calc_fast_wavespeed(u_ll, normal_direction, equations)
    c_f_rr = calc_fast_wavespeed(u_rr, normal_direction, equations)
    v_roe, c_f_roe = calc_fast_wavespeed_roe(u_ll, u_rr, normal_direction, equations)

    # Estimate the min/max eigenvalues in the normal direction
    λ_min = min(v_normal_ll - c_f_ll, v_roe - c_f_roe)
    λ_max = max(v_normal_rr + c_f_rr, v_roe + c_f_roe)

    return λ_min, λ_max
end

# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this rotation of the state vector
@inline function rotate_to_x(u, normal_vector, equations::IdealGlmMhdEquations2D)
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D rotation matrix with normal and tangent directions of the form
    # [ 1   0    0   0   0   0    0   0   0;
    #   0  n_1  n_2  0   0   0    0   0   0;
    #   0  t_1  t_2  0   0   0    0   0   0;
    #   0   0    0   1   0   0    0   0   0;
    #   0   0    0   0   1   0    0   0   0;
    #   0   0    0   0   0  n_1  n_2  0   0;
    #   0   0    0   0   0  t_1  t_2  0   0;
    #   0   0    0   0   0   0    0   1   0;
    #   0   0    0   0   0   0    0   0   1 ]
    # where t_1 = -n_2 and t_2 = n_1.
    # Note for IdealGlmMhdEquations2D only the velocities and magnetic field variables rotate

    return SVector(u[1],
                   c * u[2] + s * u[3],
                   -s * u[2] + c * u[3],
                   u[4],
                   u[5],
                   c * u[6] + s * u[7],
                   -s * u[6] + c * u[7],
                   u[8],
                   u[9])
end

# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this back-rotation of the state vector
@inline function rotate_from_x(u, normal_vector, equations::IdealGlmMhdEquations2D)
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D back-rotation matrix with normal and tangent directions of the form
    # [ 1   0    0   0   0   0    0   0   0;
    #   0  n_1  t_1  0   0   0    0   0   0;
    #   0  n_2  t_2  0   0   0    0   0   0;
    #   0   0    0   1   0   0    0   0   0;
    #   0   0    0   0   1   0    0   0   0;
    #   0   0    0   0   0  n_1  t_1  0   0;
    #   0   0    0   0   0  n_2  t_2  0   0;
    #   0   0    0   0   0   0    0   1   0;
    #   0   0    0   0   0   0    0   0   1 ]
    # where t_1 = -n_2 and t_2 = n_1.
    # Note for IdealGlmMhdEquations2D the velocities and magnetic field variables back-rotate

    return SVector(u[1],
                   c * u[2] - s * u[3],
                   s * u[2] + c * u[3],
                   u[4],
                   u[5],
                   c * u[6] - s * u[7],
                   s * u[6] + c * u[7],
                   u[8],
                   u[9])
end

@inline function max_abs_speeds(u, equations::IdealGlmMhdEquations2D)
    rho, rho_v1, rho_v2, rho_v3, _ = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    cf_x_direction = calc_fast_wavespeed(u, 1, equations)
    cf_y_direction = calc_fast_wavespeed(u, 2, equations)

    return abs(v1) + cf_x_direction, abs(v2) + cf_y_direction
end

# Convert conservative variables to primitive
@inline function cons2prim(u, equations::IdealGlmMhdEquations2D)
    rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    p = (equations.gamma - 1) * (rho_e -
         0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3
          + B1 * B1 + B2 * B2 + B3 * B3
          + psi * psi))

    return SVector(rho, v1, v2, v3, p, B1, B2, B3, psi)
end

# Convert conservative variables to entropy variables
@inline function cons2entropy(u, equations::IdealGlmMhdEquations2D)
    rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    v_square = v1^2 + v2^2 + v3^2
    p = (equations.gamma - 1) *
        (rho_e - 0.5 * rho * v_square - 0.5 * (B1^2 + B2^2 + B3^2) - 0.5 * psi^2)
    s = log(p) - equations.gamma * log(rho)
    rho_p = rho / p

    w1 = (equations.gamma - s) * equations.inv_gamma_minus_one - 0.5 * rho_p * v_square
    w2 = rho_p * v1
    w3 = rho_p * v2
    w4 = rho_p * v3
    w5 = -rho_p
    w6 = rho_p * B1
    w7 = rho_p * B2
    w8 = rho_p * B3
    w9 = rho_p * psi

    return SVector(w1, w2, w3, w4, w5, w6, w7, w8, w9)
end

# Convert entropy variables to conservative variables
@inline function entropy2cons(w, equations::IdealGlmMhdEquations2D)
    w1, w2, w3, w4, w5, w6, w7, w8, w9 = w

    v1 = -w2 / w5
    v2 = -w3 / w5
    v3 = -w4 / w5

    B1 = -w6 / w5
    B2 = -w7 / w5
    B3 = -w8 / w5
    psi = -w9 / w5

    # This imitates what is done for compressible Euler 3D `entropy2cons`: we convert from
    # the entropy variables for `-rho * s / (gamma - 1)` to the entropy variables for the entropy
    # `-rho * s` used by Hughes, Franca, Mallet (1986).
    @unpack gamma = equations
    V1, V2, V3, V4, V5 = SVector(w1, w2, w3, w4, w5) * (gamma - 1)
    s = gamma - V1 + (V2^2 + V3^2 + V4^2) / (2 * V5)
    rho_iota = ((gamma - 1) / (-V5)^gamma)^(equations.inv_gamma_minus_one) *
               exp(-s * equations.inv_gamma_minus_one)
    rho = -rho_iota * V5
    p = -rho / w5

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::IdealGlmMhdEquations2D)
    rho, v1, v2, v3, p, B1, B2, B3, psi = prim

    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_v3 = rho * v3
    rho_e = p * equations.inv_gamma_minus_one +
            0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3) +
            0.5 * (B1^2 + B2^2 + B3^2) + 0.5 * psi^2

    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi)
end

@inline function density(u, equations::IdealGlmMhdEquations2D)
    rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
    return rho
end

@inline function pressure(u, equations::IdealGlmMhdEquations2D)
    rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
    p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho
         -
         0.5 * (B1^2 + B2^2 + B3^2)
         -
         0.5 * psi^2)
    return p
end

@inline function density_pressure(u, equations::IdealGlmMhdEquations2D)
    rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
    p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho
         -
         0.5 * (B1^2 + B2^2 + B3^2)
         -
         0.5 * psi^2)
    return rho * p
end

# Compute the fastest wave speed for ideal MHD equations: c_f, the fast magnetoacoustic eigenvalue
@inline function calc_fast_wavespeed(cons, orientation::Integer,
                                     equations::IdealGlmMhdEquations2D)
    rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = cons
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    kin_en = 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3)
    mag_en = 0.5 * (B1 * B1 + B2 * B2 + B3 * B3)
    p = (equations.gamma - 1) * (rho_e - kin_en - mag_en - 0.5 * psi^2)
    a_square = equations.gamma * p / rho
    sqrt_rho = sqrt(rho)
    b1 = B1 / sqrt_rho
    b2 = B2 / sqrt_rho
    b3 = B3 / sqrt_rho
    b_square = b1 * b1 + b2 * b2 + b3 * b3
    if orientation == 1 # x-direction
        c_f = sqrt(0.5 * (a_square + b_square) +
                   0.5 * sqrt((a_square + b_square)^2 - 4.0 * a_square * b1^2))
    else
        c_f = sqrt(0.5 * (a_square + b_square) +
                   0.5 * sqrt((a_square + b_square)^2 - 4.0 * a_square * b2^2))
    end
    return c_f
end

@inline function calc_fast_wavespeed(cons, normal_direction::AbstractVector,
                                     equations::IdealGlmMhdEquations2D)
    rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = cons
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    kin_en = 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3)
    mag_en = 0.5 * (B1 * B1 + B2 * B2 + B3 * B3)
    p = (equations.gamma - 1) * (rho_e - kin_en - mag_en - 0.5 * psi^2)
    a_square = equations.gamma * p / rho
    sqrt_rho = sqrt(rho)
    b1 = B1 / sqrt_rho
    b2 = B2 / sqrt_rho
    b3 = B3 / sqrt_rho
    b_square = b1 * b1 + b2 * b2 + b3 * b3
    norm_squared = (normal_direction[1] * normal_direction[1] +
                    normal_direction[2] * normal_direction[2])
    b_dot_n_squared = (b1 * normal_direction[1] +
                       b2 * normal_direction[2])^2 / norm_squared

    c_f = sqrt((0.5 * (a_square + b_square) +
                0.5 * sqrt((a_square + b_square)^2 - 4 * a_square * b_dot_n_squared)) *
               norm_squared)
    return c_f
end

"""
    calc_fast_wavespeed_roe(u_ll, u_rr, orientation_or_normal_direction, equations::IdealGlmMhdEquations2D)

Compute the fast magnetoacoustic wave speed using Roe averages
as given by
- Cargo and Gallice (1997)
  Roe Matrices for Ideal MHD and Systematic Construction
  of Roe Matrices for Systems of Conservation Laws
  [DOI: 10.1006/jcph.1997.5773](https://doi.org/10.1006/jcph.1997.5773)
"""
@inline function calc_fast_wavespeed_roe(u_ll, u_rr, orientation::Integer,
                                         equations::IdealGlmMhdEquations2D)
    rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

    # Calculate primitive variables
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    v3_ll = rho_v3_ll / rho_ll
    kin_en_ll = 0.5 * (rho_v1_ll * v1_ll + rho_v2_ll * v2_ll + rho_v3_ll * v3_ll)
    mag_norm_ll = B1_ll * B1_ll + B2_ll * B2_ll + B3_ll * B3_ll
    p_ll = (equations.gamma - 1) *
           (rho_e_ll - kin_en_ll - 0.5 * mag_norm_ll - 0.5 * psi_ll^2)

    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    v3_rr = rho_v3_rr / rho_rr
    kin_en_rr = 0.5 * (rho_v1_rr * v1_rr + rho_v2_rr * v2_rr + rho_v3_rr * v3_rr)
    mag_norm_rr = B1_rr * B1_rr + B2_rr * B2_rr + B3_rr * B3_rr
    p_rr = (equations.gamma - 1) *
           (rho_e_rr - kin_en_rr - 0.5 * mag_norm_rr - 0.5 * psi_rr^2)

    # compute total pressure which is thermal + magnetic pressures
    p_total_ll = p_ll + 0.5 * mag_norm_ll
    p_total_rr = p_rr + 0.5 * mag_norm_rr

    # compute the Roe density averages
    sqrt_rho_ll = sqrt(rho_ll)
    sqrt_rho_rr = sqrt(rho_rr)
    inv_sqrt_rho_add = 1.0 / (sqrt_rho_ll + sqrt_rho_rr)
    inv_sqrt_rho_prod = 1.0 / (sqrt_rho_ll * sqrt_rho_rr)
    rho_ll_roe = sqrt_rho_ll * inv_sqrt_rho_add
    rho_rr_roe = sqrt_rho_rr * inv_sqrt_rho_add
    # Roe averages
    # velocities and magnetic fields
    v1_roe = v1_ll * rho_ll_roe + v1_rr * rho_rr_roe
    v2_roe = v2_ll * rho_ll_roe + v2_rr * rho_rr_roe
    v3_roe = v3_ll * rho_ll_roe + v3_rr * rho_rr_roe
    B1_roe = B1_ll * rho_ll_roe + B1_rr * rho_rr_roe
    B2_roe = B2_ll * rho_ll_roe + B2_rr * rho_rr_roe
    B3_roe = B3_ll * rho_ll_roe + B3_rr * rho_rr_roe
    # enthalpy
    H_ll = (rho_e_ll + p_total_ll) / rho_ll
    H_rr = (rho_e_rr + p_total_rr) / rho_rr
    H_roe = H_ll * rho_ll_roe + H_rr * rho_rr_roe
    # temporary variable see equation (4.12) in Cargo and Gallice
    X = 0.5 * ((B1_ll - B1_rr)^2 + (B2_ll - B2_rr)^2 + (B3_ll - B3_rr)^2) *
        inv_sqrt_rho_add^2
    # averaged components needed to compute c_f, the fast magnetoacoustic wave speed
    b_square_roe = (B1_roe^2 + B2_roe^2 + B3_roe^2) * inv_sqrt_rho_prod # scaled magnectic sum
    a_square_roe = ((2.0 - equations.gamma) * X +
                    (equations.gamma - 1.0) *
                    (H_roe - 0.5 * (v1_roe^2 + v2_roe^2 + v3_roe^2) -
                     b_square_roe)) # acoustic speed
    # finally compute the average wave speed and set the output velocity (depends on orientation)
    if orientation == 1 # x-direction
        c_a_roe = B1_roe^2 * inv_sqrt_rho_prod # (squared) Alfvén wave speed
        a_star_roe = sqrt((a_square_roe + b_square_roe)^2 -
                          4.0 * a_square_roe * c_a_roe)
        c_f_roe = sqrt(0.5 * (a_square_roe + b_square_roe + a_star_roe))
        vel_out_roe = v1_roe
    else # y-direction
        c_a_roe = B2_roe^2 * inv_sqrt_rho_prod # (squared) Alfvén wave speed
        a_star_roe = sqrt((a_square_roe + b_square_roe)^2 -
                          4.0 * a_square_roe * c_a_roe)
        c_f_roe = sqrt(0.5 * (a_square_roe + b_square_roe + a_star_roe))
        vel_out_roe = v2_roe
    end

    return vel_out_roe, c_f_roe
end

@inline function calc_fast_wavespeed_roe(u_ll, u_rr, normal_direction::AbstractVector,
                                         equations::IdealGlmMhdEquations2D)
    rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

    # Calculate primitive variables
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    v3_ll = rho_v3_ll / rho_ll
    kin_en_ll = 0.5 * (rho_v1_ll * v1_ll + rho_v2_ll * v2_ll + rho_v3_ll * v3_ll)
    mag_norm_ll = B1_ll * B1_ll + B2_ll * B2_ll + B3_ll * B3_ll
    p_ll = (equations.gamma - 1) *
           (rho_e_ll - kin_en_ll - 0.5 * mag_norm_ll - 0.5 * psi_ll^2)

    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    v3_rr = rho_v3_rr / rho_rr
    kin_en_rr = 0.5 * (rho_v1_rr * v1_rr + rho_v2_rr * v2_rr + rho_v3_rr * v3_rr)
    mag_norm_rr = B1_rr * B1_rr + B2_rr * B2_rr + B3_rr * B3_rr
    p_rr = (equations.gamma - 1) *
           (rho_e_rr - kin_en_rr - 0.5 * mag_norm_rr - 0.5 * psi_rr^2)

    # compute total pressure which is thermal + magnetic pressures
    p_total_ll = p_ll + 0.5 * mag_norm_ll
    p_total_rr = p_rr + 0.5 * mag_norm_rr

    # compute the Roe density averages
    sqrt_rho_ll = sqrt(rho_ll)
    sqrt_rho_rr = sqrt(rho_rr)
    inv_sqrt_rho_add = 1.0 / (sqrt_rho_ll + sqrt_rho_rr)
    inv_sqrt_rho_prod = 1.0 / (sqrt_rho_ll * sqrt_rho_rr)
    rho_ll_roe = sqrt_rho_ll * inv_sqrt_rho_add
    rho_rr_roe = sqrt_rho_rr * inv_sqrt_rho_add
    # Roe averages
    # velocities and magnetic fields
    v1_roe = v1_ll * rho_ll_roe + v1_rr * rho_rr_roe
    v2_roe = v2_ll * rho_ll_roe + v2_rr * rho_rr_roe
    v3_roe = v3_ll * rho_ll_roe + v3_rr * rho_rr_roe
    B1_roe = B1_ll * rho_ll_roe + B1_rr * rho_rr_roe
    B2_roe = B2_ll * rho_ll_roe + B2_rr * rho_rr_roe
    B3_roe = B3_ll * rho_ll_roe + B3_rr * rho_rr_roe
    # enthalpy
    H_ll = (rho_e_ll + p_total_ll) / rho_ll
    H_rr = (rho_e_rr + p_total_rr) / rho_rr
    H_roe = H_ll * rho_ll_roe + H_rr * rho_rr_roe
    # temporary variable see equation (4.12) in Cargo and Gallice
    X = 0.5 * ((B1_ll - B1_rr)^2 + (B2_ll - B2_rr)^2 + (B3_ll - B3_rr)^2) *
        inv_sqrt_rho_add^2
    # averaged components needed to compute c_f, the fast magnetoacoustic wave speed
    b_square_roe = (B1_roe^2 + B2_roe^2 + B3_roe^2) * inv_sqrt_rho_prod # scaled magnectic sum
    a_square_roe = ((2.0 - equations.gamma) * X +
                    (equations.gamma - 1.0) *
                    (H_roe - 0.5 * (v1_roe^2 + v2_roe^2 + v3_roe^2) -
                     b_square_roe)) # acoustic speed

    # finally compute the average wave speed and set the output velocity (depends on orientation)
    norm_squared = (normal_direction[1] * normal_direction[1] +
                    normal_direction[2] * normal_direction[2])
    B_roe_dot_n_squared = (B1_roe * normal_direction[1] +
                           B2_roe * normal_direction[2])^2 / norm_squared

    c_a_roe = B_roe_dot_n_squared * inv_sqrt_rho_prod # (squared) Alfvén wave speed
    a_star_roe = sqrt((a_square_roe + b_square_roe)^2 - 4 * a_square_roe * c_a_roe)
    c_f_roe = sqrt(0.5 * (a_square_roe + b_square_roe + a_star_roe) * norm_squared)
    vel_out_roe = (v1_roe * normal_direction[1] +
                   v2_roe * normal_direction[2])

    return vel_out_roe, c_f_roe
end

# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equations::IdealGlmMhdEquations2D)
    # Pressure
    p = (equations.gamma - 1) *
        (cons[5] - 1 / 2 * (cons[2]^2 + cons[3]^2 + cons[4]^2) / cons[1]
         -
         1 / 2 * (cons[6]^2 + cons[7]^2 + cons[8]^2)
         -
         1 / 2 * cons[9]^2)

    # Thermodynamic entropy
    s = log(p) - equations.gamma * log(cons[1])

    return s
end

# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equations::IdealGlmMhdEquations2D)
    S = -entropy_thermodynamic(cons, equations) * cons[1] *
        equations.inv_gamma_minus_one

    return S
end

# Default entropy is the mathematical entropy
@inline entropy(cons, equations::IdealGlmMhdEquations2D) = entropy_math(cons, equations)

# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::IdealGlmMhdEquations2D) = cons[5]

# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(cons, equations::IdealGlmMhdEquations2D)
    return 0.5 * (cons[2]^2 + cons[3]^2 + cons[4]^2) / cons[1]
end

# Calculate the magnetic energy for a conservative state `cons'.
#  OBS! For non-dinmensional form of the ideal MHD magnetic pressure ≡ magnetic energy
@inline function energy_magnetic(cons, ::IdealGlmMhdEquations2D)
    return 0.5 * (cons[6]^2 + cons[7]^2 + cons[8]^2)
end

# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equations::IdealGlmMhdEquations2D)
    return (energy_total(cons, equations)
            -
            energy_kinetic(cons, equations)
            -
            energy_magnetic(cons, equations)
            -
            cons[9]^2 / 2)
end

# Calculate the cross helicity (\vec{v}⋅\vec{B}) for a conservative state `cons'
@inline function cross_helicity(cons, ::IdealGlmMhdEquations2D)
    return (cons[2] * cons[6] + cons[3] * cons[7] + cons[4] * cons[8]) / cons[1]
end
end # @muladd
