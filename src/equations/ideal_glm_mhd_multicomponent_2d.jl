# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    IdealGlmMhdMulticomponentEquations2D

The ideal compressible multicomponent GLM-MHD equations in two space dimensions.
"""
struct IdealGlmMhdMulticomponentEquations2D{NVARS, NCOMP, RealT <: Real} <:
       AbstractIdealGlmMhdMulticomponentEquations{2, NVARS, NCOMP}
    gammas::SVector{NCOMP, RealT}
    gas_constants::SVector{NCOMP, RealT}
    cv::SVector{NCOMP, RealT}
    cp::SVector{NCOMP, RealT}
    c_h::RealT # GLM cleaning speed

    function IdealGlmMhdMulticomponentEquations2D{NVARS, NCOMP, RealT}(gammas::SVector{NCOMP,
                                                                                       RealT},
                                                                       gas_constants::SVector{NCOMP,
                                                                                              RealT},
                                                                       c_h::RealT) where {
                                                                                          NVARS,
                                                                                          NCOMP,
                                                                                          RealT <:
                                                                                          Real
                                                                                          }
        NCOMP >= 1 ||
            throw(DimensionMismatch("`gammas` and `gas_constants` have to be filled with at least one value"))

        cv = gas_constants ./ (gammas .- 1)
        cp = gas_constants + gas_constants ./ (gammas .- 1)

        new(gammas, gas_constants, cv, cp, c_h)
    end
end

function IdealGlmMhdMulticomponentEquations2D(; gammas, gas_constants)
    _gammas = promote(gammas...)
    _gas_constants = promote(gas_constants...)
    RealT = promote_type(eltype(_gammas), eltype(_gas_constants))

    NVARS = length(_gammas) + 8
    NCOMP = length(_gammas)

    __gammas = SVector(map(RealT, _gammas))
    __gas_constants = SVector(map(RealT, _gas_constants))

    c_h = convert(RealT, NaN)

    return IdealGlmMhdMulticomponentEquations2D{NVARS, NCOMP, RealT}(__gammas,
                                                                     __gas_constants,
                                                                     c_h)
end

# Outer constructor for `@reset` works correctly
function IdealGlmMhdMulticomponentEquations2D(gammas, gas_constants, cv, cp, c_h)
    _gammas = promote(gammas...)
    _gas_constants = promote(gas_constants...)
    RealT = promote_type(eltype(_gammas), eltype(_gas_constants))

    NVARS = length(_gammas) + 8
    NCOMP = length(_gammas)

    __gammas = SVector(map(RealT, _gammas))
    __gas_constants = SVector(map(RealT, _gas_constants))

    c_h = convert(RealT, c_h)

    return IdealGlmMhdMulticomponentEquations2D{NVARS, NCOMP, RealT}(__gammas,
                                                                     __gas_constants,
                                                                     c_h)
end

@inline function Base.real(::IdealGlmMhdMulticomponentEquations2D{NVARS, NCOMP, RealT}) where {
                                                                                               NVARS,
                                                                                               NCOMP,
                                                                                               RealT
                                                                                               }
    RealT
end

have_nonconservative_terms(::IdealGlmMhdMulticomponentEquations2D) = True()

function varnames(::typeof(cons2cons), equations::IdealGlmMhdMulticomponentEquations2D)
    cons = ("rho_v1", "rho_v2", "rho_v3", "rho_e", "B1", "B2", "B3", "psi")
    rhos = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
    return (cons..., rhos...)
end

function varnames(::typeof(cons2prim), equations::IdealGlmMhdMulticomponentEquations2D)
    prim = ("v1", "v2", "v3", "p", "B1", "B2", "B3", "psi")
    rhos = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
    return (prim..., rhos...)
end

function default_analysis_integrals(::IdealGlmMhdMulticomponentEquations2D)
    (entropy_timederivative, Val(:l2_divb), Val(:linf_divb))
end

"""
    initial_condition_convergence_test(x, t, equations::IdealGlmMhdMulticomponentEquations2D)

An Alfvén wave as smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t,
                                            equations::IdealGlmMhdMulticomponentEquations2D)
    # smooth Alfvén wave test from Derigs et al. FLASH (2016)
    # domain must be set to [0, 1/cos(α)] x [0, 1/sin(α)], γ = 5/3
    RealT = eltype(x)
    alpha = 0.25f0 * convert(RealT, pi)
    x_perp = x[1] * cos(alpha) + x[2] * sin(alpha)
    B_perp = convert(RealT, 0.1) * sinpi(2 * x_perp)
    rho = one(RealT)
    prim_rho = SVector{ncomponents(equations), real(equations)}(2^(i - 1) * (1 - 2) *
                                                                rho / (1 -
                                                                 2^ncomponents(equations))
                                                                for i in eachcomponent(equations))

    v1 = -B_perp * sin(alpha)
    v2 = B_perp * cos(alpha)
    v3 = convert(RealT, 0.1) * cospi(2 * x_perp)
    p = convert(RealT, 0.1)
    B1 = cos(alpha) + v1
    B2 = sin(alpha) + v2
    B3 = v3
    psi = 0
    prim_other = SVector(v1, v2, v3, p, B1, B2, B3, psi)

    return prim2cons(vcat(prim_other, prim_rho), equations)
end

"""
    initial_condition_weak_blast_wave(x, t, equations::IdealGlmMhdMulticomponentEquations2D)

A weak blast wave adapted from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t,
                                           equations::IdealGlmMhdMulticomponentEquations2D)
    # Adapted MHD version of the weak blast wave from Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
    # Same discontinuity in the velocities but with magnetic fields
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    prim_rho = SVector{ncomponents(equations), real(equations)}(r > 0.5f0 ?
                                                                2^(i - 1) * (1 - 2) /
                                                                (RealT(1) -
                                                                 2^ncomponents(equations)) :
                                                                2^(i - 1) * (1 - 2) *
                                                                RealT(1.1691) /
                                                                (1 -
                                                                 2^ncomponents(equations))
                                                                for i in eachcomponent(equations))

    v1 = r > 0.5f0 ? zero(RealT) : convert(RealT, 0.1882) * cos_phi
    v2 = r > 0.5f0 ? zero(RealT) : convert(RealT, 0.1882) * sin_phi
    p = r > 0.5f0 ? one(RealT) : convert(RealT, 1.245)

    prim_other = SVector(v1, v2, 0, p, 1, 1, 1, 0)

    return prim2cons(vcat(prim_other, prim_rho), equations)
end

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer,
                      equations::IdealGlmMhdMulticomponentEquations2D)
    rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
    @unpack c_h = equations

    rho = density(u, equations)

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    kin_en = 0.5f0 * rho * (v1^2 + v2^2 + v3^2)
    mag_en = 0.5f0 * (B1^2 + B2^2 + B3^2)
    gamma = totalgamma(u, equations)
    p = (gamma - 1) * (rho_e - kin_en - mag_en - 0.5f0 * psi^2)

    if orientation == 1
        f_rho = densities(u, v1, equations)
        f1 = rho_v1 * v1 + p + mag_en - B1^2
        f2 = rho_v1 * v2 - B1 * B2
        f3 = rho_v1 * v3 - B1 * B3
        f4 = (kin_en + gamma * p / (gamma - 1) + 2 * mag_en) * v1 -
             B1 * (v1 * B1 + v2 * B2 + v3 * B3) + c_h * psi * B1
        f5 = c_h * psi
        f6 = v1 * B2 - v2 * B1
        f7 = v1 * B3 - v3 * B1
        f8 = c_h * B1
    else # orientation == 2
        f_rho = densities(u, v2, equations)
        f1 = rho_v2 * v1 - B1 * B2
        f2 = rho_v2 * v2 + p + mag_en - B2^2
        f3 = rho_v2 * v3 - B2 * B3
        f4 = (kin_en + gamma * p / (gamma - 1) + 2 * mag_en) * v2 -
             B2 * (v1 * B1 + v2 * B2 + v3 * B3) + c_h * psi * B2
        f5 = v2 * B1 - v1 * B2
        f6 = c_h * psi
        f7 = v2 * B3 - v3 * B2
        f8 = c_h * B2
    end

    f_other = SVector(f1, f2, f3, f4, f5, f6, f7, f8)

    return vcat(f_other, f_rho)
end

"""
    flux_nonconservative_powell(u_ll, u_rr, orientation::Integer,
                                equations::IdealGlmMhdMulticomponentEquations2D)

Non-symmetric two-point flux discretizing the nonconservative (source) term of
Powell and the Galilean nonconservative term associated with the GLM multiplier
of the [`IdealGlmMhdMulticomponentEquations2D`](@ref).

## References
- Marvin Bohm, Andrew R.Winters, Gregor J. Gassner, Dominik Derigs,
  Florian Hindenlang, Joachim Saur
  An entropy stable nodal discontinuous Galerkin method for the resistive MHD
  equations. Part I: Theory and numerical verification
  [DOI: 10.1016/j.jcp.2018.06.027](https://doi.org/10.1016/j.jcp.2018.06.027)
"""
@inline function flux_nonconservative_powell(u_ll, u_rr, orientation::Integer,
                                             equations::IdealGlmMhdMulticomponentEquations2D)
    rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
    rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

    rho_ll = density(u_ll, equations)

    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    v3_ll = rho_v3_ll / rho_ll
    v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll

    # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
    # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
    # Note that the order of conserved variables is changed compared to the
    # standard GLM MHD equations, i.e., the densities are moved to the end
    # Here, we compute the non-density components at first and append zero density
    # components afterwards
    zero_densities = SVector{ncomponents(equations), real(equations)}(ntuple(_ -> zero(real(equations)),
                                                                             Val(ncomponents(equations))))
    if orientation == 1
        f = SVector(B1_ll * B1_rr,
                    B2_ll * B1_rr,
                    B3_ll * B1_rr,
                    v_dot_B_ll * B1_rr + v1_ll * psi_ll * psi_rr,
                    v1_ll * B1_rr,
                    v2_ll * B1_rr,
                    v3_ll * B1_rr,
                    v1_ll * psi_rr)
    else # orientation == 2
        f = SVector(B1_ll * B2_rr,
                    B2_ll * B2_rr,
                    B3_ll * B2_rr,
                    v_dot_B_ll * B2_rr + v2_ll * psi_ll * psi_rr,
                    v1_ll * B2_rr,
                    v2_ll * B2_rr,
                    v3_ll * B2_rr,
                    v2_ll * psi_rr)
    end

    return vcat(f, zero_densities)
end

"""
    flux_derigs_etal(u_ll, u_rr, orientation, equations::IdealGlmMhdMulticomponentEquations2D)

Entropy conserving two-point flux adapted by
- Derigs et al. (2018)
  Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field
  divergence diminishing ideal magnetohydrodynamics equations for multicomponent
  [DOI: 10.1016/j.jcp.2018.03.002](https://doi.org/10.1016/j.jcp.2018.03.002)
"""
function flux_derigs_etal(u_ll, u_rr, orientation::Integer,
                          equations::IdealGlmMhdMulticomponentEquations2D)
    # Unpack left and right states to get velocities, pressure, and inverse temperature (called beta)
    rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
    rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr
    @unpack gammas, gas_constants, cv, c_h = equations

    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)

    gamma_ll = totalgamma(u_ll, equations)
    gamma_rr = totalgamma(u_rr, equations)

    rhok_mean = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i + 8],
                                                                         u_rr[i + 8])
                                                                 for i in eachcomponent(equations))
    rhok_avg = SVector{ncomponents(equations), real(equations)}(0.5f0 * (u_ll[i + 8] +
                                                                 u_rr[i + 8])
                                                                for i in eachcomponent(equations))

    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    v3_ll = rho_v3_ll / rho_ll
    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    v3_rr = rho_v3_rr / rho_rr
    v1_sq = 0.5f0 * (v1_ll^2 + v1_rr^2)
    v2_sq = 0.5f0 * (v2_ll^2 + v2_rr^2)
    v3_sq = 0.5f0 * (v3_ll^2 + v3_rr^2)
    v_sq = v1_sq + v2_sq + v3_sq
    B1_sq = 0.5f0 * (B1_ll^2 + B1_rr^2)
    B2_sq = 0.5f0 * (B2_ll^2 + B2_rr^2)
    B3_sq = 0.5f0 * (B3_ll^2 + B3_rr^2)
    B_sq = B1_sq + B2_sq + B3_sq
    vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
    vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2
    mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
    mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
    # for convenience store v⋅B
    vel_dot_mag_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll
    vel_dot_mag_rr = v1_rr * B1_rr + v2_rr * B2_rr + v3_rr * B3_rr

    # Compute the necessary mean values needed for either direction
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v3_avg = 0.5f0 * (v3_ll + v3_rr)
    v_sum = v1_avg + v2_avg + v3_avg
    B1_avg = 0.5f0 * (B1_ll + B1_rr)
    B2_avg = 0.5f0 * (B2_ll + B2_rr)
    B3_avg = 0.5f0 * (B3_ll + B3_rr)
    psi_avg = 0.5f0 * (psi_ll + psi_rr)
    vel_norm_avg = 0.5f0 * (vel_norm_ll + vel_norm_rr)
    mag_norm_avg = 0.5f0 * (mag_norm_ll + mag_norm_rr)
    vel_dot_mag_avg = 0.5f0 * (vel_dot_mag_ll + vel_dot_mag_rr)

    RealT = eltype(u_ll)
    enth = zero(RealT)
    help1_ll = zero(RealT)
    help1_rr = zero(RealT)

    for i in eachcomponent(equations)
        enth += rhok_avg[i] * gas_constants[i]
        help1_ll += u_ll[i + 8] * cv[i]
        help1_rr += u_rr[i + 8] * cv[i]
    end

    T_ll = (rho_e_ll - 0.5f0 * rho_ll * (vel_norm_ll) - 0.5f0 * mag_norm_ll -
            0.5f0 * psi_ll^2) / help1_ll
    T_rr = (rho_e_rr - 0.5f0 * rho_rr * (vel_norm_rr) - 0.5f0 * mag_norm_rr -
            0.5f0 * psi_rr^2) / help1_rr
    T = 0.5f0 * (1 / T_ll + 1 / T_rr)
    T_log = ln_mean(1 / T_ll, 1 / T_rr)

    # Calculate fluxes depending on orientation with specific direction averages
    help1 = zero(RealT)
    help2 = zero(RealT)
    if orientation == 1
        f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v1_avg
                                                                 for i in eachcomponent(equations))
        for i in eachcomponent(equations)
            help1 += f_rho[i] * cv[i]
            help2 += f_rho[i]
        end
        f1 = help2 * v1_avg + enth / T + 0.5f0 * mag_norm_avg - B1_avg * B1_avg
        f2 = help2 * v2_avg - B1_avg * B2_avg
        f3 = help2 * v3_avg - B1_avg * B3_avg
        f5 = c_h * psi_avg
        f6 = v1_avg * B2_avg - v2_avg * B1_avg
        f7 = v1_avg * B3_avg - v3_avg * B1_avg
        f8 = c_h * B1_avg
        # total energy flux is complicated and involves the previous eight components
        psi_B1_avg = 0.5f0 * (B1_ll * psi_ll + B1_rr * psi_rr)
        v1_mag_avg = 0.5f0 * (v1_ll * mag_norm_ll + v1_rr * mag_norm_rr)

        f4 = (help1 / T_log) - 0.5f0 * (vel_norm_avg) * (help2) + f1 * v1_avg +
             f2 * v2_avg + f3 * v3_avg +
             f5 * B1_avg + f6 * B2_avg + f7 * B3_avg + f8 * psi_avg -
             0.5f0 * v1_mag_avg +
             B1_avg * vel_dot_mag_avg - c_h * psi_B1_avg

    else
        f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v2_avg
                                                                 for i in eachcomponent(equations))
        for i in eachcomponent(equations)
            help1 += f_rho[i] * cv[i]
            help2 += f_rho[i]
        end
        f1 = help2 * v1_avg - B1_avg * B2_avg
        f2 = help2 * v2_avg + enth / T + 0.5f0 * mag_norm_avg - B2_avg * B2_avg
        f3 = help2 * v3_avg - B2_avg * B3_avg
        f5 = v2_avg * B1_avg - v1_avg * B2_avg
        f6 = c_h * psi_avg
        f7 = v2_avg * B3_avg - v3_avg * B2_avg
        f8 = c_h * B2_avg

        # total energy flux is complicated and involves the previous eight components
        psi_B2_avg = 0.5f0 * (B2_ll * psi_ll + B2_rr * psi_rr)
        v2_mag_avg = 0.5f0 * (v2_ll * mag_norm_ll + v2_rr * mag_norm_rr)

        f4 = (help1 / T_log) - 0.5f0 * (vel_norm_avg) * (help2) + f1 * v1_avg +
             f2 * v2_avg + f3 * v3_avg +
             f5 * B1_avg + f6 * B2_avg + f7 * B3_avg + f8 * psi_avg -
             0.5f0 * v2_mag_avg +
             B2_avg * vel_dot_mag_avg - c_h * psi_B2_avg
    end

    f_other = SVector(f1, f2, f3, f4, f5, f6, f7, f8)

    return vcat(f_other, f_rho)
end

"""
    flux_hindenlang_gassner(u_ll, u_rr, orientation_or_normal_direction,
                            equations::IdealGlmMhdMulticomponentEquations2D)

Adaption of the entropy conserving and kinetic energy preserving two-point flux of
Hindenlang (2019), extending [`flux_ranocha`](@ref) to the MHD equations.
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
                                         equations::IdealGlmMhdMulticomponentEquations2D)
    # Unpack left and right states
    v1_ll, v2_ll, v3_ll, p_ll, B1_ll, B2_ll, B3_ll, psi_ll = cons2prim(u_ll, equations)
    v1_rr, v2_rr, v3_rr, p_rr, B1_rr, B2_rr, B3_rr, psi_rr = cons2prim(u_rr, equations)

    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)

    # Compute the necessary mean values needed for either direction
    # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
    # in exact arithmetic since
    #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
    #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
    inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v3_avg = 0.5f0 * (v3_ll + v3_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    psi_avg = 0.5f0 * (psi_ll + psi_rr)
    velocity_square_avg = 0.5f0 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)
    magnetic_square_avg = 0.5f0 * (B1_ll * B1_rr + B2_ll * B2_rr + B3_ll * B3_rr)

    inv_gamma_minus_one = 1 / (totalgamma(0.5f0 * (u_ll + u_rr), equations) - 1)

    rhok_mean = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i + 8],
                                                                         u_rr[i + 8])
                                                                 for i in eachcomponent(equations))
    rhok_avg = SVector{ncomponents(equations), real(equations)}(0.5f0 * (u_ll[i + 8] +
                                                                 u_rr[i + 8])
                                                                for i in eachcomponent(equations))

    RealT = eltype(u_ll)
    if orientation == 1
        f1 = zero(RealT)
        f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v1_avg
                                                                 for i in eachcomponent(equations))
        for i in eachcomponent(equations)
            f1 += f_rho[i]
        end

        # Calculate fluxes depending on orientation with specific direction averages
        f2 = f1 * v1_avg + p_avg + magnetic_square_avg -
             0.5f0 * (B1_ll * B1_rr + B1_rr * B1_ll)
        f3 = f1 * v2_avg - 0.5f0 * (B1_ll * B2_rr + B1_rr * B2_ll)
        f4 = f1 * v3_avg - 0.5f0 * (B1_ll * B3_rr + B1_rr * B3_ll)
        # f5 below
        f6 = f6 = equations.c_h * psi_avg
        f7 = 0.5f0 * (v1_ll * B2_ll - v2_ll * B1_ll + v1_rr * B2_rr - v2_rr * B1_rr)
        f8 = 0.5f0 * (v1_ll * B3_ll - v3_ll * B1_ll + v1_rr * B3_rr - v3_rr * B1_rr)
        f9 = equations.c_h * 0.5f0 * (B1_ll + B1_rr)
        # total energy flux is complicated and involves the previous components
        f5 = (f1 * (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one)
              +
              0.5f0 * (+p_ll * v1_rr + p_rr * v1_ll
               + (v1_ll * B2_ll * B2_rr + v1_rr * B2_rr * B2_ll)
               + (v1_ll * B3_ll * B3_rr + v1_rr * B3_rr * B3_ll)
               -
               (v2_ll * B1_ll * B2_rr + v2_rr * B1_rr * B2_ll)
               -
               (v3_ll * B1_ll * B3_rr + v3_rr * B1_rr * B3_ll)
               +
               equations.c_h * (B1_ll * psi_rr + B1_rr * psi_ll)))
    else
        f1 = zero(RealT)
        f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v2_avg
                                                                 for i in eachcomponent(equations))
        for i in eachcomponent(equations)
            f1 += f_rho[i]
        end

        # Calculate fluxes depending on orientation with specific direction averages
        f2 = f1 * v1_avg - 0.5f0 * (B2_ll * B1_rr + B2_rr * B1_ll)
        f3 = f1 * v2_avg + p_avg + magnetic_square_avg -
             0.5f0 * (B2_ll * B2_rr + B2_rr * B2_ll)
        f4 = f1 * v3_avg - 0.5f0 * (B2_ll * B3_rr + B2_rr * B3_ll)
        #f5 below
        f6 = 0.5f0 * (v2_ll * B1_ll - v1_ll * B2_ll + v2_rr * B1_rr - v1_rr * B2_rr)
        f7 = equations.c_h * psi_avg
        f8 = 0.5f0 * (v2_ll * B3_ll - v3_ll * B2_ll + v2_rr * B3_rr - v3_rr * B2_rr)
        f9 = equations.c_h * 0.5f0 * (B2_ll + B2_rr)
        # total energy flux is complicated and involves the previous components
        f5 = (f1 * (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one)
              +
              0.5f0 * (+p_ll * v2_rr + p_rr * v2_ll
               + (v2_ll * B1_ll * B1_rr + v2_rr * B1_rr * B1_ll)
               + (v2_ll * B3_ll * B3_rr + v2_rr * B3_rr * B3_ll)
               -
               (v1_ll * B2_ll * B1_rr + v1_rr * B2_rr * B1_ll)
               -
               (v3_ll * B2_ll * B3_rr + v3_rr * B2_rr * B3_ll)
               +
               equations.c_h * (B2_ll * psi_rr + B2_rr * psi_ll)))
    end

    f_other = SVector(f2, f3, f4, f5, f6, f7, f8, f9)

    return vcat(f_other, f_rho)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::IdealGlmMhdMulticomponentEquations2D)
    rho_v1_ll, rho_v2_ll, _ = u_ll
    rho_v1_rr, rho_v2_rr, _ = u_rr

    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)

    # Calculate velocities and fast magnetoacoustic wave speeds
    if orientation == 1
        v_ll = rho_v1_ll / rho_ll
        v_rr = rho_v1_rr / rho_rr
    else # orientation == 2
        v_ll = rho_v2_ll / rho_ll
        v_rr = rho_v2_rr / rho_rr
    end
    cf_ll = calc_fast_wavespeed(u_ll, orientation, equations)
    cf_rr = calc_fast_wavespeed(u_rr, orientation, equations)

    λ_max = max(abs(v_ll), abs(v_rr)) + max(cf_ll, cf_rr)
end

@inline function max_abs_speeds(u, equations::IdealGlmMhdMulticomponentEquations2D)
    rho_v1, rho_v2, _ = u

    rho = density(u, equations)

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho

    cf_x_direction = calc_fast_wavespeed(u, 1, equations)
    cf_y_direction = calc_fast_wavespeed(u, 2, equations)

    return (abs(v1) + cf_x_direction, abs(v2) + cf_y_direction)
end

@inline function density_pressure(u, equations::IdealGlmMhdMulticomponentEquations2D)
    rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
    rho = density(u, equations)
    gamma = totalgamma(u, equations)
    p = (gamma - 1) * (rho_e - 0.5f0 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho
         -
         0.5f0 * (B1^2 + B2^2 + B3^2)
         -
         0.5f0 * psi^2)
    return rho * p
end

# Convert conservative variables to primitive
function cons2prim(u, equations::IdealGlmMhdMulticomponentEquations2D)
    rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u

    prim_rho = SVector{ncomponents(equations), real(equations)}(u[i + 8]
                                                                for i in eachcomponent(equations))
    rho = density(u, equations)

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho

    gamma = totalgamma(u, equations)

    p = (gamma - 1) *
        (rho_e - 0.5f0 * rho * (v1^2 + v2^2 + v3^2) - 0.5f0 * (B1^2 + B2^2 + B3^2) -
         0.5f0 * psi^2)
    prim_other = SVector(v1, v2, v3, p, B1, B2, B3, psi)

    return vcat(prim_other, prim_rho)
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::IdealGlmMhdMulticomponentEquations2D)
    rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
    @unpack cv, gammas, gas_constants = equations

    rho = density(u, equations)

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    v_square = v1^2 + v2^2 + v3^2
    gamma = totalgamma(u, equations)
    p = (gamma - 1) *
        (rho_e - 0.5f0 * rho * v_square - 0.5f0 * (B1^2 + B2^2 + B3^2) - 0.5f0 * psi^2)
    s = log(p) - gamma * log(rho)
    rho_p = rho / p

    # Multicomponent stuff
    help1 = zero(v1)

    for i in eachcomponent(equations)
        help1 += u[i + 8] * cv[i]
    end

    T = (rho_e - 0.5f0 * rho * v_square - 0.5f0 * (B1^2 + B2^2 + B3^2) - 0.5f0 * psi^2) /
        (help1)

    entrop_rho = SVector{ncomponents(equations), real(equations)}(-1 *
                                                                  (cv[i] * log(T) -
                                                                   gas_constants[i] *
                                                                   log(u[i + 8])) +
                                                                  gas_constants[i] +
                                                                  cv[i] -
                                                                  (v_square / (2 * T))
                                                                  for i in eachcomponent(equations))

    w1 = v1 / T
    w2 = v2 / T
    w3 = v3 / T
    w4 = -1 / T
    w5 = B1 / T
    w6 = B2 / T
    w7 = B3 / T
    w8 = psi / T

    entrop_other = SVector(w1, w2, w3, w4, w5, w6, w7, w8)

    return vcat(entrop_other, entrop_rho)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::IdealGlmMhdMulticomponentEquations2D)
    v1, v2, v3, p, B1, B2, B3, psi = prim

    cons_rho = SVector{ncomponents(equations), real(equations)}(prim[i + 8]
                                                                for i in eachcomponent(equations))
    rho = density(prim, equations)

    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_v3 = rho * v3

    gamma = totalgamma(prim, equations)
    rho_e = p / (gamma - 1) + 0.5f0 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3) +
            0.5f0 * (B1^2 + B2^2 + B3^2) + 0.5f0 * psi^2

    cons_other = SVector(rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3,
                         psi)

    return vcat(cons_other, cons_rho)
end

# Compute the fastest wave speed for ideal MHD equations: c_f, the fast magnetoacoustic eigenvalue
@inline function calc_fast_wavespeed(cons, direction,
                                     equations::IdealGlmMhdMulticomponentEquations2D)
    rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = cons
    rho = density(cons, equations)
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    v_mag = sqrt(v1^2 + v2^2 + v3^2)
    gamma = totalgamma(cons, equations)
    p = (gamma - 1) *
        (rho_e - 0.5f0 * rho * v_mag^2 - 0.5f0 * (B1^2 + B2^2 + B3^2) - 0.5f0 * psi^2)
    a_square = gamma * p / rho
    sqrt_rho = sqrt(rho)
    b1 = B1 / sqrt_rho
    b2 = B2 / sqrt_rho
    b3 = B3 / sqrt_rho
    b_square = b1^2 + b2^2 + b3^2
    if direction == 1 # x-direction
        c_f = sqrt(0.5f0 * (a_square + b_square) +
                   0.5f0 * sqrt((a_square + b_square)^2 - 4 * a_square * b1^2))
    else
        c_f = sqrt(0.5f0 * (a_square + b_square) +
                   0.5f0 * sqrt((a_square + b_square)^2 - 4 * a_square * b2^2))
    end
    return c_f
end

@inline function density(u, equations::IdealGlmMhdMulticomponentEquations2D)
    RealT = eltype(u)
    rho = zero(RealT)

    for i in eachcomponent(equations)
        rho += u[i + 8]
    end

    return rho
end

@inline function totalgamma(u, equations::IdealGlmMhdMulticomponentEquations2D)
    @unpack cv, gammas = equations

    RealT = eltype(u)
    help1 = zero(RealT)
    help2 = zero(RealT)

    for i in eachcomponent(equations)
        help1 += u[i + 8] * cv[i] * gammas[i]
        help2 += u[i + 8] * cv[i]
    end

    return help1 / help2
end

@inline function densities(u, v, equations::IdealGlmMhdMulticomponentEquations2D)
    return SVector{ncomponents(equations), real(equations)}(u[i + 8] * v
                                                            for i in eachcomponent(equations))
end
end # @muladd
