# This file includes functions that are common for all AbstractIdealGlmMhdMultiIonEquations

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

have_nonconservative_terms(::AbstractIdealGlmMhdMultiIonEquations) = True()

function varnames(::typeof(cons2cons), equations::AbstractIdealGlmMhdMultiIonEquations)
    cons = ("B1", "B2", "B3")
    for i in eachcomponent(equations)
        cons = (cons...,
                tuple("rho_" * string(i), "rho_v1_" * string(i), "rho_v2_" * string(i),
                      "rho_v3_" * string(i), "rho_e_" * string(i))...)
    end
    cons = (cons..., "psi")

    return cons
end

function varnames(::typeof(cons2prim), equations::AbstractIdealGlmMhdMultiIonEquations)
    prim = ("B1", "B2", "B3")
    for i in eachcomponent(equations)
        prim = (prim...,
                tuple("rho_" * string(i), "v1_" * string(i), "v2_" * string(i),
                      "v3_" * string(i), "p_" * string(i))...)
    end
    prim = (prim..., "psi")

    return prim
end

function default_analysis_integrals(::AbstractIdealGlmMhdMultiIonEquations)
    (entropy_timederivative, Val(:l2_divb), Val(:linf_divb))
end

"""
    source_terms_lorentz(u, x, t, equations::AbstractIdealGlmMhdMultiIonEquations)

Source terms due to the Lorentz' force for plasmas with more than one ion species. These source 
terms are a fundamental, inseparable part of the multi-ion GLM-MHD equations, and vanish for 
a single-species plasma. In particular, they have to be used for every
simulation of [`IdealGlmMhdMultiIonEquations2D`](@ref).
"""
function source_terms_lorentz(u, x, t, equations::AbstractIdealGlmMhdMultiIonEquations)
    @unpack charge_to_mass = equations
    B1, B2, B3 = magnetic_field(u, equations)
    v1_plus, v2_plus, v3_plus, vk1_plus, vk2_plus, vk3_plus = charge_averaged_velocities(u,
                                                                                         equations)

    s = zero(MVector{nvariables(equations), eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
        rho_inv = 1 / rho
        v1 = rho_v1 * rho_inv
        v2 = rho_v2 * rho_inv
        v3 = rho_v3 * rho_inv
        v1_diff = v1_plus - v1
        v2_diff = v2_plus - v2
        v3_diff = v3_plus - v3
        r_rho = charge_to_mass[k] * rho
        s2 = r_rho * (v2_diff * B3 - v3_diff * B2)
        s3 = r_rho * (v3_diff * B1 - v1_diff * B3)
        s4 = r_rho * (v1_diff * B2 - v2_diff * B1)
        s5 = v1 * s2 + v2 * s3 + v3 * s4

        set_component!(s, k, 0, s2, s3, s4, s5, equations)
    end

    return SVector(s)
end

"""
    electron_pressure_zero(u, equations::AbstractIdealGlmMhdMultiIonEquations)

Returns the value of zero for the electron pressure. Needed for consistency with the 
single-fluid MHD equations in the limit of one ion species.
"""
function electron_pressure_zero(u, equations::AbstractIdealGlmMhdMultiIonEquations)
    return zero(u[1])
end

"""
    v1, v2, v3, vk1, vk2, vk3 = charge_averaged_velocities(u,
                                                       equations::AbstractIdealGlmMhdMultiIonEquations)


Compute the charge-averaged velocities (`v1`, `v2`, and `v3`) and each ion species' contribution
to the charge-averaged velocities (`vk1`, `vk2`, and `vk3`). The output variables `vk1`, `vk2`, and `vk3`
are `SVectors` of size `ncomponents(equations)`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function charge_averaged_velocities(u,
                                            equations::AbstractIdealGlmMhdMultiIonEquations)
    total_electron_charge = zero(real(equations))

    vk1_plus = zero(MVector{ncomponents(equations), eltype(u)})
    vk2_plus = zero(MVector{ncomponents(equations), eltype(u)})
    vk3_plus = zero(MVector{ncomponents(equations), eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, _ = get_component(k, u, equations)

        total_electron_charge += rho * equations.charge_to_mass[k]
        vk1_plus[k] = rho_v1 * equations.charge_to_mass[k]
        vk2_plus[k] = rho_v2 * equations.charge_to_mass[k]
        vk3_plus[k] = rho_v3 * equations.charge_to_mass[k]
    end
    vk1_plus ./= total_electron_charge
    vk2_plus ./= total_electron_charge
    vk3_plus ./= total_electron_charge
    v1_plus = sum(vk1_plus)
    v2_plus = sum(vk2_plus)
    v3_plus = sum(vk3_plus)

    return v1_plus, v2_plus, v3_plus, SVector(vk1_plus), SVector(vk2_plus),
           SVector(vk3_plus)
end

"""
    get_component(k, u, equations::AbstractIdealGlmMhdMultiIonEquations)

Get the hydrodynamic variables of component (ion species) `k`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function get_component(k, u, equations::AbstractIdealGlmMhdMultiIonEquations)
    return SVector(u[3 + (k - 1) * 5 + 1],
                   u[3 + (k - 1) * 5 + 2],
                   u[3 + (k - 1) * 5 + 3],
                   u[3 + (k - 1) * 5 + 4],
                   u[3 + (k - 1) * 5 + 5])
end

"""
    set_component!(u, k, u1, u2, u3, u4, u5,
                   equations::AbstractIdealGlmMhdMultiIonEquations)

Set the hydrodynamic variables (`u1` to `u5`) of component (ion species) `k`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function set_component!(u, k, u1, u2, u3, u4, u5,
                                equations::AbstractIdealGlmMhdMultiIonEquations)
    u[3 + (k - 1) * 5 + 1] = u1
    u[3 + (k - 1) * 5 + 2] = u2
    u[3 + (k - 1) * 5 + 3] = u3
    u[3 + (k - 1) * 5 + 4] = u4
    u[3 + (k - 1) * 5 + 5] = u5

    return u
end

# Extract magnetic field from solution vector
magnetic_field(u, equations::AbstractIdealGlmMhdMultiIonEquations) = SVector(u[1], u[2],
                                                                             u[3])

# Extract GLM divergence-cleaning field from solution vector
divergence_cleaning_field(u, equations::AbstractIdealGlmMhdMultiIonEquations) = u[end]

# Get total density as the sum of the individual densities of the ion species
@inline function density(u, equations::AbstractIdealGlmMhdMultiIonEquations)
    rho = zero(real(equations))
    for k in eachcomponent(equations)
        rho += u[3 + (k - 1) * 5 + 1]
    end
    return rho
end

@inline function pressure(u, equations::AbstractIdealGlmMhdMultiIonEquations)
    B1, B2, B3, _ = u
    p = zero(MVector{ncomponents(equations), real(equations)})
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        v_mag = sqrt(v1^2 + v2^2 + v3^2)
        gamma = equations.gammas[k]
        p[k] = (gamma - 1) *
               (rho_e - 0.5f0 * rho * v_mag^2 - 0.5f0 * (B1^2 + B2^2 + B3^2))
    end
    return SVector{ncomponents(equations), real(equations)}(p)
end

#Convert conservative variables to primitive
function cons2prim(u, equations::AbstractIdealGlmMhdMultiIonEquations)
    @unpack gammas = equations
    B1, B2, B3 = magnetic_field(u, equations)
    psi = divergence_cleaning_field(u, equations)

    prim = zero(MVector{nvariables(equations), eltype(u)})
    prim[1] = B1
    prim[2] = B2
    prim[3] = B3
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
        rho_inv = 1 / rho
        v1 = rho_inv * rho_v1
        v2 = rho_inv * rho_v2
        v3 = rho_inv * rho_v3

        p = (gammas[k] - 1) * (rho_e -
             0.5f0 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3
              + B1 * B1 + B2 * B2 + B3 * B3
              + psi * psi))

        set_component!(prim, k, rho, v1, v2, v3, p, equations)
    end
    prim[end] = psi

    return SVector(prim)
end

#Convert conservative variables to entropy variables
@inline function cons2entropy(u, equations::AbstractIdealGlmMhdMultiIonEquations)
    @unpack gammas = equations
    B1, B2, B3 = magnetic_field(u, equations)
    psi = divergence_cleaning_field(u, equations)

    prim = cons2prim(u, equations)
    entropy = zero(MVector{nvariables(equations), eltype(u)})
    rho_p_plus = zero(real(equations))
    for k in eachcomponent(equations)
        rho, v1, v2, v3, p = get_component(k, prim, equations)
        s = log(p) - gammas[k] * log(rho)
        rho_p = rho / p
        w1 = (gammas[k] - s) / (gammas[k] - 1) - 0.5f0 * rho_p * (v1^2 + v2^2 + v3^2)
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
    entropy[end] = rho_p_plus * psi

    return SVector(entropy)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::AbstractIdealGlmMhdMultiIonEquations)
    @unpack gammas = equations
    B1, B2, B3 = magnetic_field(prim, equations)
    psi = divergence_cleaning_field(prim, equations)

    cons = zero(MVector{nvariables(equations), eltype(prim)})
    cons[1] = B1
    cons[2] = B2
    cons[3] = B3
    for k in eachcomponent(equations)
        rho, v1, v2, v3, p = get_component(k, prim, equations)
        rho_v1 = rho * v1
        rho_v2 = rho * v2
        rho_v3 = rho * v3

        rho_e = p / (gammas[k] - 1) +
                0.5f0 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3) +
                0.5f0 * (B1^2 + B2^2 + B3^2) +
                0.5f0 * psi^2

        set_component!(cons, k, rho, rho_v1, rho_v2, rho_v3, rho_e, equations)
    end
    cons[end] = psi

    return SVector(cons)
end

@doc raw"""
    source_terms_collision_ion_ion(u, x, t,
                                   equations::AbstractIdealGlmMhdMultiIonEquations)

Compute the ion-ion collision source terms for the momentum and energy equations of each ion species as
```math
\begin{aligned}
  \vec{s}_{\rho_k \vec{v}_k} =&  \rho_k\sum_{k'}\bar{\nu}_{kk'}(\vec{v}_{k'} - \vec{v}_k),\\
  s_{E_k}  =& 
    3 \sum_{k'} \left(
    \bar{\nu}_{kk'} \frac{\rho_k M_1}{M_{k'} + M_k} R_1 (T_{k'} - T_k)
    \right) + 
    \sum_{k'} \left(
        \bar{\nu}_{kk'} \rho_k \frac{M_{k'}}{M_{k'} + M_k} \|\vec{v}_{k'} - \vec{v}_k\|^2
        \right)
        +
        \vec{v}_k \cdot \vec{s}_{\rho_k \vec{v}_k},
\end{aligned}
```
where ``M_k`` is the molar mass of ion species `k` provided in `equations.molar_masses`, 
``R_k`` is the specific gas constant of ion species `k` provided in `equations.gas_constants`, and
 ``\bar{\nu}_{kk'}`` is the effective collision frequency of species `k` with species `k'`, which is computed as
```math
\begin{aligned}
  \bar{\nu}_{kk'} = \bar{\nu}^1_{kk'} \tilde{B}_{kk'} \frac{\rho_{k'}}{T_{k k'}^{3/2}},
\end{aligned}
```
with the so-called reduced temperature ``T_{k k'}`` and the ion-ion collision constants ``\tilde{B}_{kk'}`` provided
in `equations.ion_electron_collision_constants` (see [`IdealGlmMhdMultiIonEquations2D`](@ref)).

The additional coefficient ``\bar{\nu}^1_{kk'}`` is a non-dimensional drift correction factor proposed by Rambo and 
Denavit.

References:
- P. Rambo, J. Denavit, Interpenetration and ion separation in colliding plasmas, Physics of Plasmas 1 (1994) 4050–4060.
- Schunk, R. W., Nagy, A. F. (2000). Ionospheres: Physics, plasma physics, and chemistry. 
  Cambridge university press.
"""
function source_terms_collision_ion_ion(u, x, t,
                                        equations::AbstractIdealGlmMhdMultiIonEquations)
    s = zero(MVector{nvariables(equations), eltype(u)})
    @unpack gas_constants, molar_masses, ion_ion_collision_constants = equations

    prim = cons2prim(u, equations)

    for k in eachcomponent(equations)
        rho_k, v1_k, v2_k, v3_k, p_k = get_component(k, prim, equations)
        T_k = p_k / (rho_k * gas_constants[k])

        S_q1 = zero(eltype(u))
        S_q2 = zero(eltype(u))
        S_q3 = zero(eltype(u))
        S_E = zero(eltype(u))
        for l in eachcomponent(equations)
            # Do not compute collisions of an ion species with itself
            k == l && continue

            rho_l, v1_l, v2_l, v3_l, p_l = get_component(l, prim, equations)
            T_l = p_l / (rho_l * gas_constants[l])

            # Reduced temperature
            T_kl = (molar_masses[l] * T_k + molar_masses[k] * T_l) /
                   (molar_masses[k] + molar_masses[l])

            delta_v2 = (v1_l - v1_k)^2 + (v2_l - v2_k)^2 + (v3_l - v3_k)^2

            # Compute collision frequency without drifting correction
            v_kl = ion_ion_collision_constants[k, l] * rho_l / T_kl^(3 / 2)

            # Correct the collision frequency with the drifting effect
            z2 = delta_v2 / (p_l / rho_l + p_k / rho_k)
            v_kl /= (1 + (2 / (9 * pi))^(1 / 3) * z2)^(3 / 2)

            S_q1 += rho_k * v_kl * (v1_l - v1_k)
            S_q2 += rho_k * v_kl * (v2_l - v2_k)
            S_q3 += rho_k * v_kl * (v3_l - v3_k)

            S_E += (3 * molar_masses[1] * gas_constants[1] * (T_l - T_k)
                    +
                    molar_masses[l] * delta_v2) * v_kl * rho_k /
                   (molar_masses[k] + molar_masses[l])
        end

        S_E += (v1_k * S_q1 + v2_k * S_q2 + v3_k * S_q3)

        set_component!(s, k, 0, S_q1, S_q2, S_q3, S_E, equations)
    end
    return SVector{nvariables(equations), real(equations)}(s)
end

@doc raw"""
    source_terms_collision_ion_electron(u, x, t,
                                        equations::AbstractIdealGlmMhdMultiIonEquations)

Compute the ion-electron collision source terms for the momentum and energy equations of each ion species. We assume ``v_e = v^+`` 
(no effect of currents on the electron velocity).

The collision sources read as
```math
\begin{aligned}
    \vec{s}_{\rho_k \vec{v}_k} =&  \rho_k \bar{\nu}_{ke} (\vec{v}_{e} - \vec{v}_k),
    \\
    s_{E_k}  =& 
    3  \left(
    \bar{\nu}_{ke} \frac{\rho_k M_{1}}{M_k} R_1 (T_{e} - T_k)
    \right) 
        +
        \vec{v}_k \cdot \vec{s}_{\rho_k \vec{v}_k},
\end{aligned}
```
where ``T_e`` is the electron temperature computed with the function `equations.electron_temperature`, 
``M_k`` is the molar mass of ion species `k` provided in `equations.molar_masses`, 
``R_k`` is the specific gas constant of ion species `k` provided in `equations.gas_constants`, and
``\bar{\nu}_{kk'}`` is the collision frequency of species `k` with the electrons, which is computed as
```math
\begin{aligned}
  \bar{\nu}_{ke} = \tilde{B}_{ke} \frac{e n_e}{T_e^{3/2}},
\end{aligned}
```
with the total electron charge ``e n_e`` (computed assuming quasi-neutrality), and the
ion-electron collision coefficient ``\tilde{B}_{ke}`` provided in `equations.ion_electron_collision_constants`,
which is scaled with the elementary charge (see [`IdealGlmMhdMultiIonEquations2D`](@ref)).

References:
- P. Rambo, J. Denavit, Interpenetration and ion separation in colliding plasmas, Physics of Plasmas 1 (1994) 4050–4060.
- Schunk, R. W., Nagy, A. F. (2000). Ionospheres: Physics, plasma physics, and chemistry. 
  Cambridge university press.
"""
function source_terms_collision_ion_electron(u, x, t,
                                             equations::AbstractIdealGlmMhdMultiIonEquations)
    s = zero(MVector{nvariables(equations), eltype(u)})
    @unpack gas_constants, molar_masses, ion_electron_collision_constants, electron_temperature = equations

    prim = cons2prim(u, equations)
    T_e = electron_temperature(u, equations)
    T_e32 = T_e^(3 / 2)

    v1_plus, v2_plus, v3_plus, vk1_plus, vk2_plus, vk3_plus = charge_averaged_velocities(u,
                                                                                         equations)

    # Compute total electron charge
    total_electron_charge = zero(real(equations))
    for k in eachcomponent(equations)
        rho, _ = get_component(k, u, equations)
        total_electron_charge += rho * equations.charge_to_mass[k]
    end

    for k in eachcomponent(equations)
        rho_k, v1_k, v2_k, v3_k, p_k = get_component(k, prim, equations)
        T_k = p_k / (rho_k * gas_constants[k])

        # Compute effective collision frequency
        v_ke = ion_electron_collision_constants[k] * total_electron_charge / T_e32

        S_q1 = rho_k * v_ke * (v1_plus - v1_k)
        S_q2 = rho_k * v_ke * (v2_plus - v2_k)
        S_q3 = rho_k * v_ke * (v3_plus - v3_k)

        S_E = 3 * molar_masses[1] * gas_constants[1] * (T_e - T_k) * v_ke * rho_k /
              molar_masses[k]

        S_E += (v1_k * S_q1 + v2_k * S_q2 + v3_k * S_q3)

        set_component!(s, k, 0, S_q1, S_q2, S_q3, S_E, equations)
    end
    return SVector{nvariables(equations), real(equations)}(s)
end
end
