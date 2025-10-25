# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    IdealMhdMultiIonEquations1D

The ideal compressible multi-ion MHD equations in one space dimension.
"""
mutable struct IdealMhdMultiIonEquations1D{NVARS, NCOMP, RealT <: Real} <:
               AbstractIdealGlmMhdMultiIonEquations{1, NVARS, NCOMP}
    gammas         :: SVector{NCOMP, RealT} # Heat capacity ratios
    charge_to_mass :: SVector{NCOMP, RealT} # Charge to mass ratios

    function IdealMhdMultiIonEquations1D{NVARS, NCOMP,
                                         RealT}(gammas
                                                ::SVector{NCOMP, RealT},
                                                charge_to_mass
                                                ::SVector{NCOMP, RealT}) where
             {NVARS, NCOMP, RealT <: Real}
        NCOMP >= 1 ||
            throw(DimensionMismatch("`gammas` and `charge_to_mass` have to be filled with at least one value"))

        new(gammas, charge_to_mass)
    end
end

function IdealMhdMultiIonEquations1D(; gammas, charge_to_mass)
    _gammas = promote(gammas...)
    _charge_to_mass = promote(charge_to_mass...)
    RealT = promote_type(eltype(_gammas), eltype(_charge_to_mass))

    NVARS = length(_gammas) * 5 + 3
    NCOMP = length(_gammas)

    __gammas = SVector(map(RealT, _gammas))
    __charge_to_mass = SVector(map(RealT, _charge_to_mass))

    return IdealMhdMultiIonEquations1D{NVARS, NCOMP, RealT}(__gammas, __charge_to_mass)
end

@inline function Base.real(::IdealMhdMultiIonEquations1D{NVARS, NCOMP, RealT}) where {
                                                                                      NVARS,
                                                                                      NCOMP,
                                                                                      RealT
                                                                                      }
    RealT
end

function varnames(::typeof(cons2cons), equations::IdealMhdMultiIonEquations1D)
    cons = ("B1", "B2", "B3")
    for i in eachcomponent(equations)
        cons = (cons...,
                tuple("rho_" * string(i), "rho_v1_" * string(i), "rho_v2_" * string(i),
                      "rho_v3_" * string(i), "rho_e_" * string(i))...)
    end

    return cons
end

function varnames(::typeof(cons2prim), equations::IdealMhdMultiIonEquations1D)
    prim = ("B1", "B2", "B3")
    for i in eachcomponent(equations)
        prim = (prim...,
                tuple("rho_" * string(i), "v1_" * string(i), "v2_" * string(i),
                      "v3_" * string(i), "p_" * string(i))...)
    end

    return prim
end

function default_analysis_integrals(::AbstractIdealGlmMhdMultiIonEquations)
    (entropy_timederivative,)
end

"""
    initial_condition_weak_blast_wave(x, t, equations::IdealMhdMultiIonEquations1D)

A weak blast wave adapted from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t, equations::IdealMhdMultiIonEquations1D)
    # Adapted MHD version of the weak blast wave from Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
    # Same discontinuity in the velocities but with magnetic fields
    # Set up polar coordinates
    inicenter = (0)
    x_norm = x[1] - inicenter[1]
    r = sqrt(x_norm^2)
    phi = atan(x_norm)

    # Calculate primitive variables
    rho = zero(real(equations))
    if r > 0.5
        rho = 1.0
    else
        rho = 1.1691
    end
    v1 = r > 0.5 ? 0.0 : 0.1882 * cos(phi)
    p = r > 0.5 ? 1.0 : 1.245

    prim = zero(MVector{nvariables(equations), real(equations)})
    prim[1] = 1.0
    prim[2] = 1.0
    prim[3] = 1.0
    for k in eachcomponent(equations)
        set_component!(prim, k,
                       2^(k - 1) * (1 - 2) / (1 - 2^ncomponents(equations)) * rho, v1,
                       0, 0, p, equations)
    end

    return prim2cons(SVector(prim), equations)
end

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::IdealMhdMultiIonEquations1D)
    B1, B2, B3 = magnetic_field(u, equations)

    v1_plus, v2_plus, v3_plus, vk1_plus, vk2_plus, vk3_plus = charge_averaged_velocities(u,
                                                                                         equations)

    mag_en = 0.5 * (B1^2 + B2^2 + B3^2)

    f = zero(MVector{nvariables(equations), eltype(u)})
    f[1] = 0.0
    f[2] = v1_plus * B2 - v2_plus * B1
    f[3] = v1_plus * B3 - v3_plus * B1

    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        kin_en = 0.5 * rho * (v1^2 + v2^2 + v3^2)

        gamma = equations.gammas[k]
        p = (gamma - 1) * (rho_e - kin_en - mag_en)

        f1 = rho_v1
        f2 = rho_v1 * v1 + p
        f3 = rho_v1 * v2
        f4 = rho_v1 * v3
        f5 = (kin_en + gamma * p / (gamma - 1)) * v1 + 2 * mag_en * vk1_plus[k] -
             B1 * (vk1_plus[k] * B1 + vk2_plus[k] * B2 + vk3_plus[k] * B3)

        set_component!(f, k, f1, f2, f3, f4, f5, equations)
    end

    return SVector(f)
end

"""
Total entropy-conserving non-conservative two-point "flux"" as described in 
- Rueda-Ramírez et al. (2023)
The term is composed of three parts
* The Powell term: Only needed in 1D for non-constant B1 (TODO)
* The MHD term: Implemented without the electron pressure (TODO).
* The "term 3": Implemented
"""
@inline function flux_nonconservative_ruedaramirez_etal(u_ll, u_rr,
                                                        orientation::Integer,
                                                        equations::IdealMhdMultiIonEquations1D)
    @unpack charge_to_mass = equations
    # Unpack left and right states to get the magnetic field
    B1_ll, B2_ll, B3_ll = magnetic_field(u_ll, equations)
    B1_rr, B2_rr, B3_rr = magnetic_field(u_rr, equations)

    # Compute important averages
    B1_avg = 0.5 * (B1_ll + B1_rr)
    B2_avg = 0.5 * (B2_ll + B2_rr)
    B3_avg = 0.5 * (B3_ll + B3_rr)
    mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
    mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
    mag_norm_avg = 0.5 * (mag_norm_ll + mag_norm_rr)

    # Compute charge ratio of u_ll
    charge_ratio_ll = zero(MVector{ncomponents(equations), eltype(u_ll)})
    total_electron_charge = zero(eltype(u_ll))
    for k in eachcomponent(equations)
        rho_k = u_ll[3 + (k - 1) * 5 + 1]
        charge_ratio_ll[k] = rho_k * charge_to_mass[k]
        total_electron_charge += charge_ratio_ll[k]
    end
    charge_ratio_ll ./= total_electron_charge

    # Compute auxiliary variables
    v1_plus_ll, v2_plus_ll, v3_plus_ll, vk1_plus_ll, vk2_plus_ll, vk3_plus_ll = charge_averaged_velocities(u_ll,
                                                                                                           equations)
    v1_plus_rr, v2_plus_rr, v3_plus_rr, vk1_plus_rr, vk2_plus_rr, vk3_plus_rr = charge_averaged_velocities(u_rr,
                                                                                                           equations)

    f = zero(MVector{nvariables(equations), eltype(u_ll)})
    # TODO: Add entries of Powell term for induction equation
    for k in eachcomponent(equations)
        # Compute Powell (only needed for non-constant B1)
        # TODO

        # Compute term 2 (MHD)
        # TODO: Add electron pressure term
        f2 = charge_ratio_ll[k] * (0.5 * mag_norm_avg - B1_avg * B1_avg) # + pe_mean)
        f3 = charge_ratio_ll[k] * (-B1_avg * B2_avg)
        f4 = charge_ratio_ll[k] * (-B1_avg * B3_avg)
        f5 = zero(eltype(u_ll)) # TODO! charge_ratio_ll[k] * pe_mean

        # Compute term 3 (only needed for NCOMP>1)
        vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
        vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
        vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
        vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
        vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
        vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
        vk1_minus_avg = 0.5 * (vk1_minus_ll + vk1_minus_rr)
        vk2_minus_avg = 0.5 * (vk2_minus_ll + vk2_minus_rr)
        vk3_minus_avg = 0.5 * (vk3_minus_ll + vk3_minus_rr)
        f5 += B2_ll * (vk1_minus_avg * B2_avg - vk2_minus_avg * B1_avg) +
              B3_ll * (vk1_minus_avg * B3_avg - vk3_minus_avg * B1_avg)

        # Adjust non-conservative term to Trixi discretization: CHANGE!!
        f2 = 2 * f2 - charge_ratio_ll[k] * (0.5 * mag_norm_ll - B1_ll * B1_ll)
        f3 = 2 * f3 + charge_ratio_ll[k] * B1_ll * B2_ll
        f4 = 2 * f4 + charge_ratio_ll[k] * B1_ll * B3_ll
        f5 = 2 * f5 - B2_ll * (vk1_minus_ll * B2_ll - vk2_minus_ll * B1_ll) -
             B3_ll * (vk1_minus_ll * B3_ll - vk3_minus_ll * B1_ll)

        # Append to the flux vector
        set_component!(f, k, 0, f2, f3, f4, f5, equations)
    end

    return SVector(f)
end

"""
Total central non-conservative two-point "flux"", where the symmetric parts are computed with standard averages
The term is composed of three parts
* The Powell term: Only needed in 1D for non-constant B1 (TODO). The central Powell "flux" is equivalent to the EC Powell "flux".
* The MHD term: Implemented without the electron pressure (TODO).
* The "term 3": Implemented
"""
@inline function flux_nonconservative_central(u_ll, u_rr, orientation::Integer,
                                              equations::IdealMhdMultiIonEquations1D)
    @unpack charge_to_mass = equations
    # Unpack left and right states to get the magnetic field
    B1_ll, B2_ll, B3_ll = magnetic_field(u_ll, equations)
    B1_rr, B2_rr, B3_rr = magnetic_field(u_rr, equations)

    # Compute important averages
    mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2

    # Compute charge ratio of u_ll
    charge_ratio_ll = zero(MVector{ncomponents(equations), eltype(u_ll)})
    total_electron_charge = zero(real(equations))
    for k in eachcomponent(equations)
        rho_k = u_ll[3 + (k - 1) * 5 + 1]
        charge_ratio_ll[k] = rho_k * charge_to_mass[k]
        total_electron_charge += charge_ratio_ll[k]
    end
    charge_ratio_ll ./= total_electron_charge

    # Compute auxiliary variables
    v1_plus_rr, v2_plus_rr, v3_plus_rr, vk1_plus_rr, vk2_plus_rr, vk3_plus_rr = charge_averaged_velocities(u_rr,
                                                                                                           equations)

    f = zero(MVector{nvariables(equations), eltype(u_ll)})
    # TODO: Add entries of Powell term for induction equation
    for k in eachcomponent(equations)
        # Compute Powell (only needed for non-constant B1)
        # TODO

        # Compute term 2 (MHD)
        # TODO: Add electron pressure term
        f2 = charge_ratio_ll[k] * (0.5 * mag_norm_rr - B1_rr * B1_rr) # + pe_mean)
        f3 = charge_ratio_ll[k] * (-B1_rr * B2_rr)
        f4 = charge_ratio_ll[k] * (-B1_rr * B3_rr)
        f5 = zero(eltype(u_ll)) # TODO! charge_ratio_ll[k] * pe_mean

        # Compute term 3 (only needed for NCOMP>1)
        vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
        vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
        vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
        f5 += (B2_ll * (vk1_minus_rr * B2_rr - vk2_minus_rr * B1_rr) +
               B3_ll * (vk1_minus_rr * B3_rr - vk3_minus_rr * B1_rr))

        # It's not needed to adjust to Trixi's non-conservative form

        # Append to the flux vector
        set_component!(f, k, 0, f2, f3, f4, f5, equations)
    end

    return SVector(f)
end

"""
flux_ruedaramirez_etal(u_ll, u_rr, orientation, equations::IdealMhdMultiIonEquations1D)

Entropy conserving two-point flux adapted by:
- Rueda-Ramírez et al. (2023)
This flux (together with the MHD non-conservative term) is consistent in the case of one species with the flux of:
- Derigs et al. (2018)
  Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field
  divergence diminishing ideal magnetohydrodynamics equations for multi-ion
  [DOI: 10.1016/j.jcp.2018.03.002](https://doi.org/10.1016/j.jcp.2018.03.002)
"""
function flux_ruedaramirez_etal(u_ll, u_rr, orientation::Integer,
                                equations::IdealMhdMultiIonEquations1D)
    @unpack gammas = equations
    # Unpack left and right states to get the magnetic field
    B1_ll, B2_ll, B3_ll = magnetic_field(u_ll, equations)
    B1_rr, B2_rr, B3_rr = magnetic_field(u_rr, equations)

    v1_plus_ll, v2_plus_ll, v3_plus_ll, vk1_plus_ll, vk2_plus_ll, vk3_plus_ll = charge_averaged_velocities(u_ll,
                                                                                                           equations)
    v1_plus_rr, v2_plus_rr, v3_plus_rr, vk1_plus_rr, vk2_plus_rr, vk3_plus_rr = charge_averaged_velocities(u_rr,
                                                                                                           equations)

    # Compute averages for global variables
    v1_plus_avg = 0.5 * (v1_plus_ll + v1_plus_rr)
    v2_plus_avg = 0.5 * (v2_plus_ll + v2_plus_rr)
    v3_plus_avg = 0.5 * (v3_plus_ll + v3_plus_rr)
    B1_avg = 0.5 * (B1_ll + B1_rr)
    B2_avg = 0.5 * (B2_ll + B2_rr)
    B3_avg = 0.5 * (B3_ll + B3_rr)
    mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
    mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
    mag_norm_avg = 0.5 * (mag_norm_ll + mag_norm_rr)

    f = zero(MVector{nvariables(equations), eltype(u_ll)})

    # Magnetic field components from f^MHD
    f6 = zero(eltype(u_ll))
    f7 = v1_plus_avg * B2_avg - v2_plus_avg * B1_avg
    f8 = v1_plus_avg * B3_avg - v3_plus_avg * B1_avg

    # Start building the flux
    f[1] = f6
    f[2] = f7
    f[3] = f8

    # Iterate over all components
    for k in eachcomponent(equations)
        # Unpack left and right states
        rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll = get_component(k, u_ll,
                                                                          equations)
        rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr = get_component(k, u_rr,
                                                                          equations)

        v1_ll = rho_v1_ll / rho_ll
        v2_ll = rho_v2_ll / rho_ll
        v3_ll = rho_v3_ll / rho_ll
        v1_rr = rho_v1_rr / rho_rr
        v2_rr = rho_v2_rr / rho_rr
        v3_rr = rho_v3_rr / rho_rr
        vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
        vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2

        p_ll = (gammas[k] - 1) *
               (rho_e_ll - 0.5 * rho_ll * vel_norm_ll - 0.5 * mag_norm_ll)
        p_rr = (gammas[k] - 1) *
               (rho_e_rr - 0.5 * rho_rr * vel_norm_rr - 0.5 * mag_norm_rr)
        beta_ll = 0.5 * rho_ll / p_ll
        beta_rr = 0.5 * rho_rr / p_rr
        # for convenience store vk_plus⋅B
        vel_dot_mag_ll = vk1_plus_ll[k] * B1_ll + vk2_plus_ll[k] * B2_ll +
                         vk3_plus_ll[k] * B3_ll
        vel_dot_mag_rr = vk1_plus_rr[k] * B1_rr + vk2_plus_rr[k] * B2_rr +
                         vk3_plus_rr[k] * B3_rr

        # Compute the necessary mean values needed for either direction
        rho_avg = 0.5 * (rho_ll + rho_rr)
        rho_mean = ln_mean(rho_ll, rho_rr)
        beta_mean = ln_mean(beta_ll, beta_rr)
        beta_avg = 0.5 * (beta_ll + beta_rr)
        p_mean = 0.5 * rho_avg / beta_avg
        v1_avg = 0.5 * (v1_ll + v1_rr)
        v2_avg = 0.5 * (v2_ll + v2_rr)
        v3_avg = 0.5 * (v3_ll + v3_rr)
        vel_norm_avg = 0.5 * (vel_norm_ll + vel_norm_rr)
        vel_dot_mag_avg = 0.5 * (vel_dot_mag_ll + vel_dot_mag_rr)
        vk1_plus_avg = 0.5 * (vk1_plus_ll[k] + vk1_plus_rr[k])
        vk2_plus_avg = 0.5 * (vk2_plus_ll[k] + vk2_plus_rr[k])
        vk3_plus_avg = 0.5 * (vk3_plus_ll[k] + vk3_plus_rr[k])
        # v_minus
        vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
        vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
        vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
        vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
        vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
        vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
        vk1_minus_avg = 0.5 * (vk1_minus_ll + vk1_minus_rr)
        vk2_minus_avg = 0.5 * (vk2_minus_ll + vk2_minus_rr)
        vk3_minus_avg = 0.5 * (vk3_minus_ll + vk3_minus_rr)

        # Ignore orientation since it is always "1" in 1D
        f1 = rho_mean * v1_avg
        f2 = f1 * v1_avg + p_mean
        f3 = f1 * v2_avg
        f4 = f1 * v3_avg

        # total energy flux is complicated and involves the previous eight components
        v1_plus_mag_avg = 0.5 *
                          (vk1_plus_ll[k] * mag_norm_ll + vk1_plus_rr[k] * mag_norm_rr)
        # Euler part
        f5 = f1 * 0.5 * (1 / (gammas[k] - 1) / beta_mean - vel_norm_avg) + f2 * v1_avg +
             f3 * v2_avg + f4 * v3_avg
        # MHD part
        f5 += (f6 * B1_avg + f7 * B2_avg + f8 * B3_avg - 0.5 * v1_plus_mag_avg +
               B1_avg * vel_dot_mag_avg                                               # Same terms as in Derigs (but with v_plus)
               + 0.5 * vk1_plus_avg * mag_norm_avg - vk1_plus_avg * B1_avg * B1_avg -
               vk2_plus_avg * B1_avg * B2_avg - vk3_plus_avg * B1_avg * B3_avg   # Additional terms coming from the MHD non-conservative term (momentum eqs)
               -
               B2_avg * (vk1_minus_avg * B2_avg - vk2_minus_avg * B1_avg) -
               B3_avg * (vk1_minus_avg * B3_avg - vk3_minus_avg * B1_avg))             # Terms coming from the non-conservative term 3 (induction equation!)

        set_component!(f, k, f1, f2, f3, f4, f5, equations)
    end

    return SVector(f)
end

"""
# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
  !!!ATTENTION: This routine is provisional. TODO: Update with the right max_abs_speed
"""
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::IdealMhdMultiIonEquations1D)
    # Calculate fast magnetoacoustic wave speeds
    # left
    cf_ll = calc_fast_wavespeed(u_ll, orientation, equations)
    # right
    cf_rr = calc_fast_wavespeed(u_rr, orientation, equations)

    # Calculate velocities (ignore orientation since it is always "1" in 1D)
    v_ll = zero(eltype(u_ll))
    v_rr = zero(eltype(u_rr))
    for k in eachcomponent(equations)
        rho, rho_v1, _ = get_component(k, u_ll, equations)
        v_ll = max(v_ll, abs(rho_v1 / rho))
        rho, rho_v1, _ = get_component(k, u_rr, equations)
        v_rr = max(v_rr, abs(rho_v1 / rho))
    end

    λ_max = max(abs(v_ll), abs(v_rr)) + max(cf_ll, cf_rr)
end

@inline function max_abs_speeds(u, equations::IdealMhdMultiIonEquations1D)
    v1 = zero(real(equations))
    for k in eachcomponent(equations)
        rho, rho_v1, _ = get_component(k, u, equations)
        v1 = max(v1, abs(rho_v1 / rho))
    end

    cf_x_direction = calc_fast_wavespeed(u, 1, equations)

    return (abs(v1) + cf_x_direction,)
end

"""
Convert conservative variables to primitive
"""
function cons2prim(u, equations::IdealMhdMultiIonEquations1D)
    @unpack gammas = equations
    B1, B2, B3 = magnetic_field(u, equations)

    prim = zero(MVector{nvariables(equations), eltype(u)})
    prim[1] = B1
    prim[2] = B2
    prim[3] = B3
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
        srho = 1 / rho
        v1 = srho * rho_v1
        v2 = srho * rho_v2
        v3 = srho * rho_v3

        p = (gammas[k] - 1) * (rho_e -
             0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3
              + B1 * B1 + B2 * B2 + B3 * B3))

        set_component!(prim, k, rho, v1, v2, v3, p, equations)
    end

    return SVector(prim)
end

"""
Convert conservative variables to entropy
"""
@inline function cons2entropy(u, equations::IdealMhdMultiIonEquations1D)
    @unpack gammas = equations
    B1, B2, B3 = magnetic_field(u, equations)

    prim = cons2prim(u, equations)
    entropy = zero(MVector{nvariables(equations), eltype(u)})
    rho_p_plus = zero(real(equations))
    for k in eachcomponent(equations)
        rho, v1, v2, v3, p = get_component(k, prim, equations)
        s = log(p) - gammas[k] * log(rho)
        rho_p = rho / p
        w1 = (gammas[k] - s) / (gammas[k] - 1) - 0.5 * rho_p * (v1^2 + v2^2 + v3^2)
        w2 = rho_p * v1
        w3 = rho_p * v2
        w4 = rho_p * v3
        w5 = -rho_p
        rho_p_plus += rho_p

        set_component!(entropy, k, w1, w2, w3, w4, w5, equations)
    end

    # Additional non-conservative variables
    entropy[1] = rho_p_plus * B1
    entropy[2] = rho_p_plus * B2
    entropy[3] = rho_p_plus * B3

    return SVector(entropy)
end

"""
Convert primitive to conservative variables
"""
@inline function prim2cons(prim, equations::IdealMhdMultiIonEquations1D)
    @unpack gammas = equations
    B1, B2, B3 = magnetic_field(prim, equations)

    cons = zero(MVector{nvariables(equations), eltype(prim)})
    cons[1] = B1
    cons[2] = B2
    cons[3] = B3
    for k in eachcomponent(equations)
        rho, v1, v2, v3, p = get_component(k, prim, equations)
        rho_v1 = rho * v1
        rho_v2 = rho * v2
        rho_v3 = rho * v3

        rho_e = p / (gammas[k] - 1.0) +
                0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3) +
                0.5 * (B1^2 + B2^2 + B3^2)

        set_component!(cons, k, rho, rho_v1, rho_v2, rho_v3, rho_e, equations)
    end

    return SVector{nvariables(equations), real(equations)}(cons)
end

"""
Compute the fastest wave speed for ideal MHD equations: c_f, the fast magnetoacoustic eigenvalue
  !!! ATTENTION: This routine is provisional.. Change once the fastest wave speed is known!!
"""
@inline function calc_fast_wavespeed(cons, direction,
                                     equations::IdealMhdMultiIonEquations1D)
    B1, B2, B3 = magnetic_field(cons, equations)

    c_f = zero(real(equations))
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, cons, equations)

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        v_mag = sqrt(v1^2 + v2^2 + v3^2)
        gamma = equations.gammas[k]
        p = (gamma - 1) * (rho_e - 0.5 * rho * v_mag^2 - 0.5 * (B1^2 + B2^2 + B3^2))
        a_square = gamma * p / rho
        sqrt_rho = sqrt(rho)

        b1 = B1 / sqrt_rho
        b2 = B2 / sqrt_rho
        b3 = B3 / sqrt_rho
        b_square = b1^2 + b2^2 + b3^2

        c_f = max(c_f,
                  sqrt(0.5 * (a_square + b_square) +
                       0.5 * sqrt((a_square + b_square)^2 - 4.0 * a_square * b1^2)))
    end

    return c_f
end
end # @muladd
