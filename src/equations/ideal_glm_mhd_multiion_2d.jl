# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    IdealGlmMhdMultiIonEquations2D(; gammas, charge_to_mass, 
                                   gas_constants = zero(SVector{length(gammas),
                                                                eltype(gammas)}),
                                   molar_masses = zero(SVector{length(gammas),
                                                               eltype(gammas)}),
                                   ion_ion_collision_constants = zeros(eltype(gammas),
                                                               length(gammas),
                                                               length(gammas)),
                                   ion_electron_collision_constants = zero(SVector{length(gammas),
                                                                                   eltype(gammas)}),
                                   electron_pressure = electron_pressure_zero,
                                   electron_temperature = electron_pressure_zero,
                                   initial_c_h = NaN)

The ideal compressible multi-ion MHD equations in two space dimensions augmented with a 
generalized Langange multipliers (GLM) divergence-cleaning technique. This is a
multi-species variant of the ideal GLM-MHD equations for calorically perfect plasmas
with independent momentum and energy equations for each ion species. This implementation 
assumes that the equations are non-dimensionalized such, that the vacuum permeability is ``\mu_0 = 1``.

In case of more than one ion species, the specific heat capacity ratios `gammas` and the charge-to-mass 
ratios `charge_to_mass` should be passed as tuples, e.g., `gammas=(1.4, 1.667)`.

The ion-ion and ion-electron collision source terms can be computed using the functions 
[`source_terms_collision_ion_ion`](@ref) and [`source_terms_collision_ion_electron`](@ref), respectively.

For ion-ion collision terms, the optional keyword arguments `gas_constants`, `molar_masses`, and `ion_ion_collision_constants` 
must be provided.  For ion-electron collision terms, the optional keyword arguments `gas_constants`, `molar_masses`, 
`ion_electron_collision_constants`, and `electron_temperature` are required.

- **`gas_constants`** and **`molar_masses`** are tuples containing the gas constant and molar mass of each 
  ion species, respectively. The **molar masses** can be provided in any unit system, as they are only used to 
  compute ratios and are independent of the other arguments.

- **`ion_ion_collision_constants`** is a symmetric matrix that contains coefficients to compute the collision
  frequencies between pairs of ion species. For example, `ion_ion_collision_constants[2, 3]` contains the collision 
  coefficient for collisions between the ion species 2 and the ion species 3. These constants are derived using the kinetic
  theory of gases (see, e.g., *Schunk & Nagy, 2000*). They are related to the collision coefficients ``B_{st}`` listed
  in Table 4.3 of *Schunk & Nagy (2000)*, but are scaled by the molecular mass of ion species ``t`` (i.e., 
  `ion_ion_collision_constants[2, 3] = ` ``B_{st}/m_{t}``) and must be provided in consistent physical units 
  (Schunk & Nagy use ``cm^3 K^{3/2} / s``). 
  See [`source_terms_collision_ion_ion`](@ref) for more details on how these constants are used to compute the collision
  frequencies.

- **`ion_electron_collision_constants`** is a tuple containing coefficients to compute the ion-electron collision frequency 
  for each ion species. They correspond to the collision coefficients `B_{se}` divided by the elementary charge. 
  The ion-electron collision frequencies can also be computed using the kinetic theory 
  of gases (see, e.g., *Schunk & Nagy, 2000*). See [`source_terms_collision_ion_electron`](@ref) for more details on how these
  constants are used to compute the collision frequencies.

- **`electron_temperature`** is a function with the signature `electron_temperature(u, equations)` that can be used
  compute the electron temperature as a function of the state `u`. The electron temperature is relevant for the computation 
  of the ion-electron collision source terms.

The argument `electron_pressure` can be used to pass a function that computes the electron
pressure as a function of the state `u` with the signature `electron_pressure(u, equations)`.
The gradient of the electron pressure is relevant for the computation of the Lorentz flux
and non-conservative term. By default, the electron pressure is zero.

The argument `initial_c_h` can be used to set the GLM divergence-cleaning speed. Note that 
`initial_c_h = 0` deactivates the divergence cleaning. The callback [`GlmSpeedCallback`](@ref)
can be used to adjust the GLM divergence-cleaning speed according to the time-step size.

References:
- G. Toth, A. Glocer, Y. Ma, D. Najib, Multi-Ion Magnetohydrodynamics 429 (2010). Numerical 
  Modeling of Space Plasma Flows, 213–218.
- A. Rueda-Ramírez, A. Sikstel, G. Gassner, An Entropy-Stable Discontinuous Galerkin Discretization
  of the Ideal Multi-Ion Magnetohydrodynamics System (2024). Journal of Computational Physics.
  [DOI: 10.1016/j.jcp.2024.113655](https://doi.org/10.1016/j.jcp.2024.113655).
- Schunk, R. W., & Nagy, A. F. (2000). Ionospheres: Physics, plasma physics, and chemistry. 
  Cambridge university press.

!!! info "The multi-ion GLM-MHD equations require source terms"
    In case of more than one ion species, the multi-ion GLM-MHD equations should ALWAYS be used
    with [`source_terms_lorentz`](@ref).
"""
mutable struct IdealGlmMhdMultiIonEquations2D{NVARS, NCOMP, RealT <: Real,
                                              ElectronPressure, ElectronTemperature} <:
               AbstractIdealGlmMhdMultiIonEquations{2, NVARS, NCOMP}
    gammas::SVector{NCOMP, RealT} # Heat capacity ratios
    charge_to_mass::SVector{NCOMP, RealT} # Charge to mass ratios
    gas_constants::SVector{NCOMP, RealT} # Specific gas constants
    molar_masses::SVector{NCOMP, RealT} # Molar masses (can be provided in any units as they are only used to compute ratios)
    ion_ion_collision_constants::Array{RealT, 2} # Symmetric matrix of collision frequency coefficients
    ion_electron_collision_constants::SVector{NCOMP, RealT} # Constants for the ion-electron collision frequencies. The collision frequency is obtained as constant * (e * n_e) / T_e^1.5
    electron_pressure::ElectronPressure # Function to compute the electron pressure
    electron_temperature::ElectronTemperature # Function to compute the electron temperature
    c_h::RealT # GLM cleaning speed

    function IdealGlmMhdMultiIonEquations2D{NVARS, NCOMP, RealT, ElectronPressure,
                                            ElectronTemperature}(gammas
                                                                 ::SVector{NCOMP,
                                                                           RealT},
                                                                 charge_to_mass
                                                                 ::SVector{NCOMP,
                                                                           RealT},
                                                                 gas_constants
                                                                 ::SVector{NCOMP,
                                                                           RealT},
                                                                 molar_masses
                                                                 ::SVector{NCOMP,
                                                                           RealT},
                                                                 ion_ion_collision_constants
                                                                 ::Array{RealT, 2},
                                                                 ion_electron_collision_constants
                                                                 ::SVector{NCOMP,
                                                                           RealT},
                                                                 electron_pressure
                                                                 ::ElectronPressure,
                                                                 electron_temperature
                                                                 ::ElectronTemperature,
                                                                 c_h::RealT) where
             {NVARS, NCOMP, RealT <: Real, ElectronPressure, ElectronTemperature}
        NCOMP >= 1 ||
            throw(DimensionMismatch("`gammas` and `charge_to_mass` have to be filled with at least one value"))

        new(gammas, charge_to_mass, gas_constants, molar_masses,
            ion_ion_collision_constants,
            ion_electron_collision_constants, electron_pressure, electron_temperature,
            c_h)
    end
end

function IdealGlmMhdMultiIonEquations2D(; gammas, charge_to_mass,
                                        gas_constants = zero(SVector{length(gammas),
                                                                     eltype(gammas)}),
                                        molar_masses = zero(SVector{length(gammas),
                                                                    eltype(gammas)}),
                                        ion_ion_collision_constants = zeros(eltype(gammas),
                                                                            length(gammas),
                                                                            length(gammas)),
                                        ion_electron_collision_constants = zero(SVector{length(gammas),
                                                                                        eltype(gammas)}),
                                        electron_pressure = electron_pressure_zero,
                                        electron_temperature = electron_pressure_zero,
                                        initial_c_h = convert(eltype(gammas), NaN))
    _gammas = promote(gammas...)
    _charge_to_mass = promote(charge_to_mass...)
    _gas_constants = promote(gas_constants...)
    _molar_masses = promote(molar_masses...)
    _ion_electron_collision_constants = promote(ion_electron_collision_constants...)
    RealT = promote_type(eltype(_gammas), eltype(_charge_to_mass),
                         eltype(_gas_constants), eltype(_molar_masses),
                         eltype(ion_ion_collision_constants),
                         eltype(_ion_electron_collision_constants))
    __gammas = SVector(map(RealT, _gammas))
    __charge_to_mass = SVector(map(RealT, _charge_to_mass))
    __gas_constants = SVector(map(RealT, _gas_constants))
    __molar_masses = SVector(map(RealT, _molar_masses))
    __ion_ion_collision_constants = map(RealT, ion_ion_collision_constants)
    __ion_electron_collision_constants = SVector(map(RealT,
                                                     _ion_electron_collision_constants))

    NVARS = length(_gammas) * 5 + 4
    NCOMP = length(_gammas)

    return IdealGlmMhdMultiIonEquations2D{NVARS, NCOMP, RealT,
                                          typeof(electron_pressure),
                                          typeof(electron_temperature)}(__gammas,
                                                                        __charge_to_mass,
                                                                        __gas_constants,
                                                                        __molar_masses,
                                                                        __ion_ion_collision_constants,
                                                                        __ion_electron_collision_constants,
                                                                        electron_pressure,
                                                                        electron_temperature,
                                                                        initial_c_h)
end

# Outer constructor for `@reset` works correctly
function IdealGlmMhdMultiIonEquations2D(gammas, charge_to_mass, gas_constants,
                                        molar_masses, ion_ion_collision_constants,
                                        ion_electron_collision_constants,
                                        electron_pressure,
                                        electron_temperature,
                                        c_h)
    return IdealGlmMhdMultiIonEquations2D(gammas = gammas,
                                          charge_to_mass = charge_to_mass,
                                          gas_constants = gas_constants,
                                          molar_masses = molar_masses,
                                          ion_ion_collision_constants = ion_ion_collision_constants,
                                          ion_electron_collision_constants = ion_electron_collision_constants,
                                          electron_pressure = electron_pressure,
                                          electron_temperature = electron_temperature,
                                          initial_c_h = c_h)
end

@inline function Base.real(::IdealGlmMhdMultiIonEquations2D{NVARS, NCOMP, RealT}) where {
                                                                                         NVARS,
                                                                                         NCOMP,
                                                                                         RealT
                                                                                         }
    RealT
end

"""
    initial_condition_weak_blast_wave(x, t, equations::IdealGlmMhdMultiIonEquations2D)

A weak blast wave (adapted to multi-ion MHD) from
- Hennemann, S., Rueda-Ramírez, A. M., Hindenlang, F. J., & Gassner, G. J. (2021). A provably entropy 
  stable subcell shock capturing approach for high order split form DG for the compressible Euler equations. 
  Journal of Computational Physics, 426, 109935. [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044).
  [DOI: 10.1016/j.jcp.2020.109935](https://doi.org/10.1016/j.jcp.2020.109935)
"""
function initial_condition_weak_blast_wave(x, t,
                                           equations::IdealGlmMhdMultiIonEquations2D)
    # Adapted MHD version of the weak blast wave from Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
    # Same discontinuity in the velocities but with magnetic fields
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = (0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)

    # Calculate primitive variables
    rho = r > 0.5f0 ? one(RealT) : convert(RealT, 1.1691)
    v1 = r > 0.5f0 ? zero(RealT) : convert(RealT, 0.1882) * cos(phi)
    v2 = r > 0.5f0 ? zero(RealT) : convert(RealT, 0.1882) * sin(phi)
    p = r > 0.5f0 ? one(RealT) : convert(RealT, 1.245)

    prim = zero(MVector{nvariables(equations), RealT})
    prim[1] = 1
    prim[2] = 1
    prim[3] = 1
    for k in eachcomponent(equations)
        set_component!(prim, k,
                       2^(k - 1) * (1 - 2) / (1 - 2^ncomponents(equations)) * rho, v1,
                       v2, 0, p, equations)
    end

    return prim2cons(SVector(prim), equations)
end

# 2D flux of the multi-ion GLM-MHD equations in the direction `orientation`
@inline function flux(u, orientation::Integer,
                      equations::IdealGlmMhdMultiIonEquations2D)
    B1, B2, B3 = magnetic_field(u, equations)
    psi = divergence_cleaning_field(u, equations)

    v1_plus, v2_plus, v3_plus, vk1_plus, vk2_plus, vk3_plus = charge_averaged_velocities(u,
                                                                                         equations)

    mag_en = 0.5f0 * (B1^2 + B2^2 + B3^2)
    div_clean_energy = 0.5f0 * psi^2

    f = zero(MVector{nvariables(equations), eltype(u)})

    if orientation == 1
        f[1] = equations.c_h * psi
        f[2] = v1_plus * B2 - v2_plus * B1
        f[3] = v1_plus * B3 - v3_plus * B1

        for k in eachcomponent(equations)
            rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
            rho_inv = 1 / rho
            v1 = rho_v1 * rho_inv
            v2 = rho_v2 * rho_inv
            v3 = rho_v3 * rho_inv
            kin_en = 0.5f0 * rho * (v1^2 + v2^2 + v3^2)

            gamma = equations.gammas[k]
            p = (gamma - 1) * (rho_e - kin_en - mag_en - div_clean_energy)

            f1 = rho_v1
            f2 = rho_v1 * v1 + p
            f3 = rho_v1 * v2
            f4 = rho_v1 * v3
            f5 = (kin_en + gamma * p / (gamma - 1)) * v1 + 2 * mag_en * vk1_plus[k] -
                 B1 * (vk1_plus[k] * B1 + vk2_plus[k] * B2 + vk3_plus[k] * B3) +
                 equations.c_h * psi * B1

            set_component!(f, k, f1, f2, f3, f4, f5, equations)
        end
        f[end] = equations.c_h * B1
    else #if orientation == 2
        f[1] = v2_plus * B1 - v1_plus * B2
        f[2] = equations.c_h * psi
        f[3] = v2_plus * B3 - v3_plus * B2

        for k in eachcomponent(equations)
            rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
            rho_inv = 1 / rho
            v1 = rho_v1 * rho_inv
            v2 = rho_v2 * rho_inv
            v3 = rho_v3 * rho_inv
            kin_en = 0.5f0 * rho * (v1^2 + v2^2 + v3^2)

            gamma = equations.gammas[k]
            p = (gamma - 1) * (rho_e - kin_en - mag_en - div_clean_energy)

            f1 = rho_v2
            f2 = rho_v2 * v1
            f3 = rho_v2 * v2 + p
            f4 = rho_v2 * v3
            f5 = (kin_en + gamma * p / (gamma - 1)) * v2 + 2 * mag_en * vk2_plus[k] -
                 B2 * (vk1_plus[k] * B1 + vk2_plus[k] * B2 + vk3_plus[k] * B3) +
                 equations.c_h * psi * B2

            set_component!(f, k, f1, f2, f3, f4, f5, equations)
        end
        f[end] = equations.c_h * B2
    end

    return SVector(f)
end

"""
    flux_nonconservative_ruedaramirez_etal(u_ll, u_rr,
                                           orientation::Integer,
                                           equations::IdealGlmMhdMultiIonEquations2D)

Entropy-conserving non-conservative two-point "flux" as described in 
- A. Rueda-Ramírez, A. Sikstel, G. Gassner, An Entropy-Stable Discontinuous Galerkin Discretization
  of the Ideal Multi-Ion Magnetohydrodynamics System (2024). Journal of Computational Physics.
  [DOI: 10.1016/j.jcp.2024.113655](https://doi.org/10.1016/j.jcp.2024.113655).

!!! info "Usage and Scaling of Non-Conservative Fluxes in Trixi.jl"
    The non-conservative fluxes derived in the reference above are written as the product
    of local and symmetric parts and are meant to be used in the same way as the conservative
    fluxes (i.e., flux + flux_noncons in both volume and surface integrals). In this routine, 
    the fluxes are multiplied by 2 because the non-conservative fluxes are always multiplied 
    by 0.5 whenever they are used in the Trixi code.

The term is composed of four individual non-conservative terms:
1. The Godunov-Powell term, which arises for plasmas with non-vanishing magnetic field divergence, and
   is evaluated as a function of the charge-averaged velocity.
2. The Lorentz-force term, which becomes a conservative term in the limit of one ion species for vanishing
   electron pressure gradients.
3. The "multi-ion" term, which vanishes in the limit of one ion species.
4. The GLM term, which is needed for Galilean invariance.
"""
@inline function flux_nonconservative_ruedaramirez_etal(u_ll, u_rr,
                                                        orientation::Integer,
                                                        equations::IdealGlmMhdMultiIonEquations2D)
    @unpack charge_to_mass = equations
    # Unpack left and right states to get the magnetic field
    B1_ll, B2_ll, B3_ll = magnetic_field(u_ll, equations)
    B1_rr, B2_rr, B3_rr = magnetic_field(u_rr, equations)
    psi_ll = divergence_cleaning_field(u_ll, equations)
    psi_rr = divergence_cleaning_field(u_rr, equations)

    # Compute important averages
    B1_avg = 0.5f0 * (B1_ll + B1_rr)
    B2_avg = 0.5f0 * (B2_ll + B2_rr)
    B3_avg = 0.5f0 * (B3_ll + B3_rr)
    mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
    mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
    mag_norm_avg = 0.5f0 * (mag_norm_ll + mag_norm_rr)
    psi_avg = 0.5f0 * (psi_ll + psi_rr)

    # Mean electron pressure
    pe_ll = equations.electron_pressure(u_ll, equations)
    pe_rr = equations.electron_pressure(u_rr, equations)
    pe_mean = 0.5f0 * (pe_ll + pe_rr)

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

    if orientation == 1
        # Entries of Godunov-Powell term for induction equation (multiply by 2 because the non-conservative flux is 
        # multiplied by 0.5 whenever it's used in the Trixi code)
        f[1] = 2 * v1_plus_ll * B1_avg
        f[2] = 2 * v2_plus_ll * B1_avg
        f[3] = 2 * v3_plus_ll * B1_avg

        for k in eachcomponent(equations)
            # Compute term Lorentz term
            f2 = charge_ratio_ll[k] * (0.5f0 * mag_norm_avg - B1_avg * B1_avg + pe_mean)
            f3 = charge_ratio_ll[k] * (-B1_avg * B2_avg)
            f4 = charge_ratio_ll[k] * (-B1_avg * B3_avg)
            f5 = vk1_plus_ll[k] * pe_mean

            # Compute multi-ion term (vanishes for NCOMP==1)
            vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
            vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
            vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            vk1_minus_avg = 0.5f0 * (vk1_minus_ll + vk1_minus_rr)
            vk2_minus_avg = 0.5f0 * (vk2_minus_ll + vk2_minus_rr)
            vk3_minus_avg = 0.5f0 * (vk3_minus_ll + vk3_minus_rr)
            f5 += (B2_ll * (vk1_minus_avg * B2_avg - vk2_minus_avg * B1_avg) +
                   B3_ll * (vk1_minus_avg * B3_avg - vk3_minus_avg * B1_avg))

            # Compute Godunov-Powell term
            f2 += charge_ratio_ll[k] * B1_ll * B1_avg
            f3 += charge_ratio_ll[k] * B2_ll * B1_avg
            f4 += charge_ratio_ll[k] * B3_ll * B1_avg
            f5 += (v1_plus_ll * B1_ll + v2_plus_ll * B2_ll + v3_plus_ll * B3_ll) *
                  B1_avg

            # Compute GLM term for the energy
            f5 += v1_plus_ll * psi_ll * psi_avg

            # Add to the flux vector (multiply by 2 because the non-conservative flux is 
            # multiplied by 0.5 whenever it's used in the Trixi code)
            set_component!(f, k, 0, 2 * f2, 2 * f3, 2 * f4, 2 * f5,
                           equations)
        end
        # Compute GLM term for psi (multiply by 2 because the non-conservative flux is 
        # multiplied by 0.5 whenever it's used in the Trixi code)
        f[end] = 2 * v1_plus_ll * psi_avg

    else #if orientation == 2
        # Entries of Godunov-Powell term for induction equation (multiply by 2 because the non-conservative flux is 
        # multiplied by 0.5 whenever it's used in the Trixi code)
        f[1] = 2 * v1_plus_ll * B2_avg
        f[2] = 2 * v2_plus_ll * B2_avg
        f[3] = 2 * v3_plus_ll * B2_avg

        for k in eachcomponent(equations)
            # Compute term Lorentz term
            f2 = charge_ratio_ll[k] * (-B2_avg * B1_avg)
            f3 = charge_ratio_ll[k] *
                 (-B2_avg * B2_avg + 0.5f0 * mag_norm_avg + pe_mean)
            f4 = charge_ratio_ll[k] * (-B2_avg * B3_avg)
            f5 = vk2_plus_ll[k] * pe_mean

            # Compute multi-ion term (vanishes for NCOMP==1)
            vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
            vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
            vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            vk1_minus_avg = 0.5f0 * (vk1_minus_ll + vk1_minus_rr)
            vk2_minus_avg = 0.5f0 * (vk2_minus_ll + vk2_minus_rr)
            vk3_minus_avg = 0.5f0 * (vk3_minus_ll + vk3_minus_rr)
            f5 += (B1_ll * (vk2_minus_avg * B1_avg - vk1_minus_avg * B2_avg) +
                   B3_ll * (vk2_minus_avg * B3_avg - vk3_minus_avg * B2_avg))

            # Compute Godunov-Powell term
            f2 += charge_ratio_ll[k] * B1_ll * B2_avg
            f3 += charge_ratio_ll[k] * B2_ll * B2_avg
            f4 += charge_ratio_ll[k] * B3_ll * B2_avg
            f5 += (v1_plus_ll * B1_ll + v2_plus_ll * B2_ll + v3_plus_ll * B3_ll) *
                  B2_avg

            # Compute GLM term for the energy
            f5 += v2_plus_ll * psi_ll * psi_avg

            # Add to the flux vector (multiply by 2 because the non-conservative flux is 
            # multiplied by 0.5 whenever it's used in the Trixi code)
            set_component!(f, k, 0, 2 * f2, 2 * f3, 2 * f4, 2 * f5,
                           equations)
        end
        # Compute GLM term for psi (multiply by 2 because the non-conservative flux is 
        # multiplied by 0.5 whenever it's used in the Trixi code)
        f[end] = 2 * v2_plus_ll * psi_avg
    end

    return SVector(f)
end

"""
    flux_nonconservative_central(u_ll, u_rr, orientation::Integer,
                                 equations::IdealGlmMhdMultiIonEquations2D)

Central non-conservative two-point "flux", where the symmetric parts are computed with standard averages.
The use of this term together with [`flux_central`](@ref) 
with [`VolumeIntegralFluxDifferencing`](@ref) yields a "standard"
(weak-form) DGSEM discretization of the multi-ion GLM-MHD system. This flux can also be used to construct a
standard local Lax-Friedrichs flux using `surface_flux = (flux_lax_friedrichs, flux_nonconservative_central)`.

!!! info "Usage and Scaling of Non-Conservative Fluxes in Trixi"
    The central non-conservative fluxes implemented in this function are written as the product
    of local and symmetric parts, where the symmetric part is a standard average. These fluxes
    are meant to be used in the same way as the conservative fluxes (i.e., flux + flux_noncons 
    in both volume and surface integrals). In this routine, the fluxes are multiplied by 2 because 
    the non-conservative fluxes are always multiplied by 0.5 whenever they are used in the Trixi code.

The term is composed of four individual non-conservative terms:
1. The Godunov-Powell term, which arises for plasmas with non-vanishing magnetic field divergence, and
   is evaluated as a function of the charge-averaged velocity.
2. The Lorentz-force term, which becomes a conservative term in the limit of one ion species for vanishing
   electron pressure gradients.
3. The "multi-ion" term, which vanishes in the limit of one ion species.
4. The GLM term, which is needed for Galilean invariance.
"""
@inline function flux_nonconservative_central(u_ll, u_rr, orientation::Integer,
                                              equations::IdealGlmMhdMultiIonEquations2D)
    @unpack charge_to_mass = equations
    # Unpack left and right states to get the magnetic field
    B1_ll, B2_ll, B3_ll = magnetic_field(u_ll, equations)
    B1_rr, B2_rr, B3_rr = magnetic_field(u_rr, equations)
    psi_ll = divergence_cleaning_field(u_ll, equations)
    psi_rr = divergence_cleaning_field(u_rr, equations)

    # Compute important averages
    mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
    mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2

    # Electron pressure 
    pe_ll = equations.electron_pressure(u_ll, equations)
    pe_rr = equations.electron_pressure(u_rr, equations)

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
    v1_plus_ll, v2_plus_ll, v3_plus_ll, vk1_plus_ll, vk2_plus_ll, vk3_plus_ll = charge_averaged_velocities(u_ll,
                                                                                                           equations)
    v1_plus_rr, v2_plus_rr, v3_plus_rr, vk1_plus_rr, vk2_plus_rr, vk3_plus_rr = charge_averaged_velocities(u_rr,
                                                                                                           equations)

    f = zero(MVector{nvariables(equations), eltype(u_ll)})

    if orientation == 1
        # Entries of Godunov-Powell term for induction equation
        f[1] = v1_plus_ll * (B1_ll + B1_rr)
        f[2] = v2_plus_ll * (B1_ll + B1_rr)
        f[3] = v3_plus_ll * (B1_ll + B1_rr)
        for k in eachcomponent(equations)
            # Compute Lorentz term
            f2 = charge_ratio_ll[k] * ((0.5f0 * mag_norm_ll - B1_ll * B1_ll + pe_ll) +
                  (0.5f0 * mag_norm_rr - B1_rr * B1_rr + pe_rr))
            f3 = charge_ratio_ll[k] * ((-B1_ll * B2_ll) + (-B1_rr * B2_rr))
            f4 = charge_ratio_ll[k] * ((-B1_ll * B3_ll) + (-B1_rr * B3_rr))
            f5 = vk1_plus_ll[k] * (pe_ll + pe_rr)

            # Compute multi-ion term, which vanishes for NCOMP==1
            vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
            vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
            vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            f5 += (B2_ll * ((vk1_minus_ll * B2_ll - vk2_minus_ll * B1_ll) +
                    (vk1_minus_rr * B2_rr - vk2_minus_rr * B1_rr)) +
                   B3_ll * ((vk1_minus_ll * B3_ll - vk3_minus_ll * B1_ll) +
                    (vk1_minus_rr * B3_rr - vk3_minus_rr * B1_rr)))

            # Compute Godunov-Powell term
            f2 += charge_ratio_ll[k] * B1_ll * (B1_ll + B1_rr)
            f3 += charge_ratio_ll[k] * B2_ll * (B1_ll + B1_rr)
            f4 += charge_ratio_ll[k] * B3_ll * (B1_ll + B1_rr)
            f5 += (v1_plus_ll * B1_ll + v2_plus_ll * B2_ll + v3_plus_ll * B3_ll) *
                  (B1_ll + B1_rr)

            # Compute GLM term for the energy
            f5 += v1_plus_ll * psi_ll * (psi_ll + psi_rr)

            # Append to the flux vector
            set_component!(f, k, 0, f2, f3, f4, f5, equations)
        end
        # Compute GLM term for psi
        f[end] = v1_plus_ll * (psi_ll + psi_rr)

    else #if orientation == 2
        # Entries of Godunov-Powell term for induction equation
        f[1] = v1_plus_ll * (B2_ll + B2_rr)
        f[2] = v2_plus_ll * (B2_ll + B2_rr)
        f[3] = v3_plus_ll * (B2_ll + B2_rr)

        for k in eachcomponent(equations)
            # Compute Lorentz term
            f2 = charge_ratio_ll[k] * ((-B2_ll * B1_ll) + (-B2_rr * B1_rr))
            f3 = charge_ratio_ll[k] * ((-B2_ll * B2_ll + 0.5f0 * mag_norm_ll + pe_ll) +
                  (-B2_rr * B2_rr + 0.5f0 * mag_norm_rr + pe_rr))
            f4 = charge_ratio_ll[k] * ((-B2_ll * B3_ll) + (-B2_rr * B3_rr))
            f5 = vk2_plus_ll[k] * (pe_ll + pe_rr)

            # Compute multi-ion term (vanishes for NCOMP==1)
            vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
            vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
            vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            f5 += (B1_ll * ((vk2_minus_ll * B1_ll - vk1_minus_ll * B2_ll) +
                    (vk2_minus_rr * B1_rr - vk1_minus_rr * B2_rr)) +
                   B3_ll * ((vk2_minus_ll * B3_ll - vk3_minus_ll * B2_ll) +
                    (vk2_minus_rr * B3_rr - vk3_minus_rr * B2_rr)))

            # Compute Godunov-Powell term
            f2 += charge_ratio_ll[k] * B1_ll * (B2_ll + B2_rr)
            f3 += charge_ratio_ll[k] * B2_ll * (B2_ll + B2_rr)
            f4 += charge_ratio_ll[k] * B3_ll * (B2_ll + B2_rr)
            f5 += (v1_plus_ll * B1_ll + v2_plus_ll * B2_ll + v3_plus_ll * B3_ll) *
                  (B2_ll + B2_rr)

            # Compute GLM term for the energy
            f5 += v2_plus_ll * psi_ll * (psi_ll + psi_rr)

            # Append to the flux vector
            set_component!(f, k, 0, f2, f3, f4, f5, equations)
        end
        # Compute GLM term for psi
        f[end] = v2_plus_ll * (psi_ll + psi_rr)
    end

    return SVector(f)
end

"""
    flux_ruedaramirez_etal(u_ll, u_rr, orientation, equations::IdealGlmMhdMultiIonEquations2D)

Entropy conserving two-point flux for the multi-ion GLM-MHD equations from
- A. Rueda-Ramírez, A. Sikstel, G. Gassner, An Entropy-Stable Discontinuous Galerkin Discretization
  of the Ideal Multi-Ion Magnetohydrodynamics System (2024). Journal of Computational Physics.
  [DOI: 10.1016/j.jcp.2024.113655](https://doi.org/10.1016/j.jcp.2024.113655).

This flux (together with the MHD non-conservative term) is consistent in the case of one ion species with the flux of:
- Derigs et al. (2018). Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field
  divergence diminishing ideal magnetohydrodynamics equations for multi-ion
  [DOI: 10.1016/j.jcp.2018.03.002](https://doi.org/10.1016/j.jcp.2018.03.002)
"""
function flux_ruedaramirez_etal(u_ll, u_rr, orientation::Integer,
                                equations::IdealGlmMhdMultiIonEquations2D)
    @unpack gammas = equations
    # Unpack left and right states to get the magnetic field
    B1_ll, B2_ll, B3_ll = magnetic_field(u_ll, equations)
    B1_rr, B2_rr, B3_rr = magnetic_field(u_rr, equations)
    psi_ll = divergence_cleaning_field(u_ll, equations)
    psi_rr = divergence_cleaning_field(u_rr, equations)

    v1_plus_ll, v2_plus_ll, v3_plus_ll, vk1_plus_ll, vk2_plus_ll, vk3_plus_ll = charge_averaged_velocities(u_ll,
                                                                                                           equations)
    v1_plus_rr, v2_plus_rr, v3_plus_rr, vk1_plus_rr, vk2_plus_rr, vk3_plus_rr = charge_averaged_velocities(u_rr,
                                                                                                           equations)

    f = zero(MVector{nvariables(equations), eltype(u_ll)})

    # Compute averages for global variables
    v1_plus_avg = 0.5f0 * (v1_plus_ll + v1_plus_rr)
    v2_plus_avg = 0.5f0 * (v2_plus_ll + v2_plus_rr)
    v3_plus_avg = 0.5f0 * (v3_plus_ll + v3_plus_rr)
    B1_avg = 0.5f0 * (B1_ll + B1_rr)
    B2_avg = 0.5f0 * (B2_ll + B2_rr)
    B3_avg = 0.5f0 * (B3_ll + B3_rr)
    mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
    mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
    mag_norm_avg = 0.5f0 * (mag_norm_ll + mag_norm_rr)
    psi_avg = 0.5f0 * (psi_ll + psi_rr)

    if orientation == 1
        psi_B1_avg = 0.5f0 * (B1_ll * psi_ll + B1_rr * psi_rr)

        # Magnetic field components from f^MHD
        f6 = equations.c_h * psi_avg
        f7 = v1_plus_avg * B2_avg - v2_plus_avg * B1_avg
        f8 = v1_plus_avg * B3_avg - v3_plus_avg * B1_avg
        f9 = equations.c_h * B1_avg

        # Start building the flux
        f[1] = f6
        f[2] = f7
        f[3] = f8
        f[end] = f9

        # Iterate over all components
        for k in eachcomponent(equations)
            # Unpack left and right states
            rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll = get_component(k, u_ll,
                                                                              equations)
            rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr = get_component(k, u_rr,
                                                                              equations)
            rho_inv_ll = 1 / rho_ll
            v1_ll = rho_v1_ll * rho_inv_ll
            v2_ll = rho_v2_ll * rho_inv_ll
            v3_ll = rho_v3_ll * rho_inv_ll
            rho_inv_rr = 1 / rho_rr
            v1_rr = rho_v1_rr * rho_inv_rr
            v2_rr = rho_v2_rr * rho_inv_rr
            v3_rr = rho_v3_rr * rho_inv_rr
            vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
            vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2

            p_ll = (gammas[k] - 1) *
                   (rho_e_ll - 0.5f0 * rho_ll * vel_norm_ll - 0.5f0 * mag_norm_ll -
                    0.5f0 * psi_ll^2)
            p_rr = (gammas[k] - 1) *
                   (rho_e_rr - 0.5f0 * rho_rr * vel_norm_rr - 0.5f0 * mag_norm_rr -
                    0.5f0 * psi_rr^2)
            beta_ll = 0.5f0 * rho_ll / p_ll
            beta_rr = 0.5f0 * rho_rr / p_rr
            # for convenience store vk_plus⋅B
            vel_dot_mag_ll = vk1_plus_ll[k] * B1_ll + vk2_plus_ll[k] * B2_ll +
                             vk3_plus_ll[k] * B3_ll
            vel_dot_mag_rr = vk1_plus_rr[k] * B1_rr + vk2_plus_rr[k] * B2_rr +
                             vk3_plus_rr[k] * B3_rr

            # Compute the necessary mean values needed for either direction
            rho_avg = 0.5f0 * (rho_ll + rho_rr)
            rho_mean = ln_mean(rho_ll, rho_rr)
            beta_mean = ln_mean(beta_ll, beta_rr)
            beta_avg = 0.5f0 * (beta_ll + beta_rr)
            p_mean = 0.5f0 * rho_avg / beta_avg
            v1_avg = 0.5f0 * (v1_ll + v1_rr)
            v2_avg = 0.5f0 * (v2_ll + v2_rr)
            v3_avg = 0.5f0 * (v3_ll + v3_rr)
            vel_norm_avg = 0.5f0 * (vel_norm_ll + vel_norm_rr)
            vel_dot_mag_avg = 0.5f0 * (vel_dot_mag_ll + vel_dot_mag_rr)
            vk1_plus_avg = 0.5f0 * (vk1_plus_ll[k] + vk1_plus_rr[k])
            vk2_plus_avg = 0.5f0 * (vk2_plus_ll[k] + vk2_plus_rr[k])
            vk3_plus_avg = 0.5f0 * (vk3_plus_ll[k] + vk3_plus_rr[k])
            # v_minus
            vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
            vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
            vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            vk1_minus_avg = 0.5f0 * (vk1_minus_ll + vk1_minus_rr)
            vk2_minus_avg = 0.5f0 * (vk2_minus_ll + vk2_minus_rr)
            vk3_minus_avg = 0.5f0 * (vk3_minus_ll + vk3_minus_rr)

            # Fill the fluxes for the mass and momentum equations
            f1 = rho_mean * v1_avg
            f2 = f1 * v1_avg + p_mean
            f3 = f1 * v2_avg
            f4 = f1 * v3_avg

            # total energy flux is complicated and involves the previous eight components
            v1_plus_mag_avg = 0.5f0 * (vk1_plus_ll[k] * mag_norm_ll +
                               vk1_plus_rr[k] * mag_norm_rr)
            # Euler part
            f5 = f1 * 0.5f0 * (1 / (gammas[k] - 1) / beta_mean - vel_norm_avg) +
                 f2 * v1_avg + f3 * v2_avg + f4 * v3_avg
            # MHD part
            f5 += (f6 * B1_avg + f7 * B2_avg + f8 * B3_avg - 0.5f0 * v1_plus_mag_avg +
                   B1_avg * vel_dot_mag_avg                                               # Same terms as in Derigs (but with v_plus)
                   + f9 * psi_avg - equations.c_h * psi_B1_avg # GLM term
                   +
                   0.5f0 * vk1_plus_avg * mag_norm_avg -
                   vk1_plus_avg * B1_avg * B1_avg - vk2_plus_avg * B1_avg * B2_avg -
                   vk3_plus_avg * B1_avg * B3_avg   # Additional terms related to the Lorentz non-conservative term (momentum eqs)
                   -
                   B2_avg * (vk1_minus_avg * B2_avg - vk2_minus_avg * B1_avg) -
                   B3_avg * (vk1_minus_avg * B3_avg - vk3_minus_avg * B1_avg))             # Terms related to the multi-ion non-conservative term (induction equation!)

            set_component!(f, k, f1, f2, f3, f4, f5, equations)
        end
    else #if orientation == 2
        psi_B2_avg = 0.5f0 * (B2_ll * psi_ll + B2_rr * psi_rr)

        # Magnetic field components from f^MHD
        f6 = v2_plus_avg * B1_avg - v1_plus_avg * B2_avg
        f7 = equations.c_h * psi_avg
        f8 = v2_plus_avg * B3_avg - v3_plus_avg * B2_avg
        f9 = equations.c_h * B2_avg

        # Start building the flux
        f[1] = f6
        f[2] = f7
        f[3] = f8
        f[end] = f9

        # Iterate over all components
        for k in eachcomponent(equations)
            # Unpack left and right states
            rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll = get_component(k, u_ll,
                                                                              equations)
            rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr = get_component(k, u_rr,
                                                                              equations)

            rho_inv_ll = 1 / rho_ll
            v1_ll = rho_v1_ll * rho_inv_ll
            v2_ll = rho_v2_ll * rho_inv_ll
            v3_ll = rho_v3_ll * rho_inv_ll
            rho_inv_rr = 1 / rho_rr
            v1_rr = rho_v1_rr * rho_inv_rr
            v2_rr = rho_v2_rr * rho_inv_rr
            v3_rr = rho_v3_rr * rho_inv_rr
            vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
            vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2

            p_ll = (gammas[k] - 1) *
                   (rho_e_ll - 0.5f0 * rho_ll * vel_norm_ll - 0.5f0 * mag_norm_ll -
                    0.5f0 * psi_ll^2)
            p_rr = (gammas[k] - 1) *
                   (rho_e_rr - 0.5f0 * rho_rr * vel_norm_rr - 0.5f0 * mag_norm_rr -
                    0.5f0 * psi_rr^2)
            beta_ll = 0.5f0 * rho_ll / p_ll
            beta_rr = 0.5f0 * rho_rr / p_rr
            # for convenience store vk_plus⋅B
            vel_dot_mag_ll = vk1_plus_ll[k] * B1_ll + vk2_plus_ll[k] * B2_ll +
                             vk3_plus_ll[k] * B3_ll
            vel_dot_mag_rr = vk1_plus_rr[k] * B1_rr + vk2_plus_rr[k] * B2_rr +
                             vk3_plus_rr[k] * B3_rr

            # Compute the necessary mean values needed for either direction
            rho_avg = 0.5f0 * (rho_ll + rho_rr)
            rho_mean = ln_mean(rho_ll, rho_rr)
            beta_mean = ln_mean(beta_ll, beta_rr)
            beta_avg = 0.5f0 * (beta_ll + beta_rr)
            p_mean = 0.5f0 * rho_avg / beta_avg
            v1_avg = 0.5f0 * (v1_ll + v1_rr)
            v2_avg = 0.5f0 * (v2_ll + v2_rr)
            v3_avg = 0.5f0 * (v3_ll + v3_rr)
            vel_norm_avg = 0.5f0 * (vel_norm_ll + vel_norm_rr)
            vel_dot_mag_avg = 0.5f0 * (vel_dot_mag_ll + vel_dot_mag_rr)
            vk1_plus_avg = 0.5f0 * (vk1_plus_ll[k] + vk1_plus_rr[k])
            vk2_plus_avg = 0.5f0 * (vk2_plus_ll[k] + vk2_plus_rr[k])
            vk3_plus_avg = 0.5f0 * (vk3_plus_ll[k] + vk3_plus_rr[k])
            # v_minus
            vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
            vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
            vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            vk1_minus_avg = 0.5f0 * (vk1_minus_ll + vk1_minus_rr)
            vk2_minus_avg = 0.5f0 * (vk2_minus_ll + vk2_minus_rr)
            vk3_minus_avg = 0.5f0 * (vk3_minus_ll + vk3_minus_rr)

            # Fill the fluxes for the mass and momentum equations
            f1 = rho_mean * v2_avg
            f2 = f1 * v1_avg
            f3 = f1 * v2_avg + p_mean
            f4 = f1 * v3_avg

            # total energy flux is complicated and involves the previous eight components
            v2_plus_mag_avg = 0.5f0 * (vk2_plus_ll[k] * mag_norm_ll +
                               vk2_plus_rr[k] * mag_norm_rr)
            # Euler part
            f5 = f1 * 0.5f0 * (1 / (gammas[k] - 1) / beta_mean - vel_norm_avg) +
                 f2 * v1_avg + f3 * v2_avg + f4 * v3_avg
            # MHD part
            f5 += (f6 * B1_avg + f7 * B2_avg + f8 * B3_avg - 0.5f0 * v2_plus_mag_avg +
                   B2_avg * vel_dot_mag_avg                                               # Same terms as in Derigs (but with v_plus)
                   + f9 * psi_avg - equations.c_h * psi_B2_avg # GLM term
                   +
                   0.5f0 * vk2_plus_avg * mag_norm_avg -
                   vk1_plus_avg * B2_avg * B1_avg - vk2_plus_avg * B2_avg * B2_avg -
                   vk3_plus_avg * B2_avg * B3_avg   # Additional terms related to the Lorentz non-conservative term (momentum eqs)
                   -
                   B1_avg * (vk2_minus_avg * B1_avg - vk1_minus_avg * B2_avg) -
                   B3_avg * (vk2_minus_avg * B3_avg - vk3_minus_avg * B2_avg))             # Terms related to the multi-ion non-conservative term (induction equation!)

            set_component!(f, k, f1, f2, f3, f4, f5, equations)
        end
    end

    return SVector(f)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
# This routine approximates the maximum wave speed as sum of the maximum ion velocity 
# for all species and the maximum magnetosonic speed.
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::IdealGlmMhdMultiIonEquations2D)
    # Calculate fast magnetoacoustic wave speeds
    # left
    cf_ll = calc_fast_wavespeed(u_ll, orientation, equations)
    # right
    cf_rr = calc_fast_wavespeed(u_rr, orientation, equations)

    # Calculate velocities
    v_ll = zero(eltype(u_ll))
    v_rr = zero(eltype(u_rr))
    if orientation == 1
        for k in eachcomponent(equations)
            rho, rho_v1, _ = get_component(k, u_ll, equations)
            v_ll = max(v_ll, abs(rho_v1 / rho))
            rho, rho_v1, _ = get_component(k, u_rr, equations)
            v_rr = max(v_rr, abs(rho_v1 / rho))
        end
    else #if orientation == 2
        for k in eachcomponent(equations)
            rho, rho_v1, rho_v2, _ = get_component(k, u_ll, equations)
            v_ll = max(v_ll, abs(rho_v2 / rho))
            rho, rho_v1, rho_v2, _ = get_component(k, u_rr, equations)
            v_rr = max(v_rr, abs(rho_v2 / rho))
        end
    end

    λ_max = max(abs(v_ll), abs(v_rr)) + max(cf_ll, cf_rr)
end

@inline function max_abs_speeds(u, equations::IdealGlmMhdMultiIonEquations2D)
    v1 = zero(real(equations))
    v2 = zero(real(equations))
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, _ = get_component(k, u, equations)
        v1 = max(v1, abs(rho_v1 / rho))
        v2 = max(v2, abs(rho_v2 / rho))
    end

    cf_x_direction = calc_fast_wavespeed(u, 1, equations)
    cf_y_direction = calc_fast_wavespeed(u, 2, equations)

    return (abs(v1) + cf_x_direction, abs(v2) + cf_y_direction)
end

# Compute the fastest wave speed for ideal multi-ion GLM-MHD equations: c_f, the fast 
# magnetoacoustic eigenvalue. This routine computes the fast magnetosonic speed for each ion
# species using the single-fluid MHD expressions and approximates the multi-ion c_f as 
# the maximum of these individual magnetosonic speeds.
@inline function calc_fast_wavespeed(cons, orientation::Integer,
                                     equations::IdealGlmMhdMultiIonEquations2D)
    B1, B2, B3 = magnetic_field(cons, equations)
    psi = divergence_cleaning_field(cons, equations)

    c_f = zero(real(equations))
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, cons, equations)

        rho_inv = 1 / rho
        v1 = rho_v1 * rho_inv
        v2 = rho_v2 * rho_inv
        v3 = rho_v3 * rho_inv
        v_mag = sqrt(v1^2 + v2^2 + v3^2)
        gamma = equations.gammas[k]
        p = (gamma - 1) *
            (rho_e - 0.5f0 * rho * v_mag^2 - 0.5f0 * (B1^2 + B2^2 + B3^2) -
             0.5f0 * psi^2)
        a_square = gamma * p * rho_inv
        inv_sqrt_rho = 1 / sqrt(rho)

        b1 = B1 * inv_sqrt_rho
        b2 = B2 * inv_sqrt_rho
        b3 = B3 * inv_sqrt_rho
        b_square = b1^2 + b2^2 + b3^2

        if orientation == 1
            c_f = max(c_f,
                      sqrt(0.5f0 * (a_square + b_square) +
                           0.5f0 *
                           sqrt((a_square + b_square)^2 - 4 * a_square * b1^2)))
        else #if orientation == 2
            c_f = max(c_f,
                      sqrt(0.5f0 * (a_square + b_square) +
                           0.5f0 *
                           sqrt((a_square + b_square)^2 - 4 * a_square * b2^2)))
        end
    end

    return c_f
end
end # @muladd
