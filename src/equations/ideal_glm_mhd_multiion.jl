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
        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
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
        srho = 1 / rho
        v1 = srho * rho_v1
        v2 = srho * rho_v2
        v3 = srho * rho_v3

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
end
