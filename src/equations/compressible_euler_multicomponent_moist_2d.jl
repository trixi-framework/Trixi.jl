# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
CompressibleMoistEulerEquations2D(;gammas, gas_constants, c_p, c_v, L_00)

Similar to `CompressibleEulerMulticomponentEquations2D` but including latent heat of
vaporization
```math
...
```

Either the specific heat ratios `gammas` and the gas constants `gas_constants` in should be
passed as tuples, or the specific heats at constant volume `cv` and the specific heats at
constant pressure `cp`. The other quantities are then calculated considering a calorically
perfect gas.
"""
struct CompressibleMoistEulerEquations2D{NVARS, NCOMP, RealT <: Real} <:
       AbstractCompressibleEulerMulticomponentEquations{2, NVARS, NCOMP}
    gammas::SVector{NCOMP, RealT}
    gas_constants::SVector{NCOMP, RealT}
    c_p::SVector{NCOMP, RealT}
    c_v::SVector{NCOMP, RealT}
    L_00::RealT # latent heat of evaporation at 0 K

    function CompressibleMoistEulerEquations2D{NVARS, NCOMP, RealT}(gammas::SVector{NCOMP,
                                                                                    RealT},
                                                                    gas_constants::SVector{NCOMP,
                                                                                           RealT},
                                                                    c_p::SVector{NCOMP,
                                                                                 RealT},
                                                                    c_v::SVector{NCOMP,
                                                                                 RealT},
                                                                    L_00::RealT) where {
                                                                                        NVARS,
                                                                                        NCOMP,
                                                                                        RealT <:
                                                                                        Real
                                                                                        }
        new(gammas, gas_constants, c_p, c_v, L_00)
    end
end

function CompressibleMoistEulerEquations2D(;
                                           gammas = nothing, gas_constants = nothing,
                                           c_p = nothing, c_v = nothing, L_00)
    if gammas !== nothing && gas_constants !== nothing
        _gammas = promote(gammas...)
        _gas_constants = promote(gas_constants...)
        RealT = promote_type(eltype(_gammas), eltype(_gas_constants),
                             typeof(gas_constants[1] / (gammas[1] - 1)))
        _c_v = _gas_constants ./ (_gammas .- 1)
        _c_p = _gas_constants + _gas_constants ./ (_gammas .- 1)
    elseif c_p !== nothing && c_v !== nothing
        _c_p = promote(c_p...)
        _c_v = promote(c_v...)
        RealT = promote_type(eltype(_c_p), eltype(_c_v), typeof(c_p[1] / c_v[1]))
        _gas_constants = _c_p .- _c_v
        _gammas = _c_p ./ _c_v
    else
        throw(DimensionMismatch("Either `gammas` and `gas_constants` or `c_p` and `c_v` \
                                 have to be filled with at least one value"))
    end

    NVARS = length(_gammas) + 3
    NCOMP = length(_gammas)

    __gammas = SVector(map(RealT, _gammas))
    __gas_constants = SVector(map(RealT, _gas_constants))
    __c_p = SVector(map(RealT, _c_p))
    __c_v = SVector(map(RealT, _c_v))

    return CompressibleMoistEulerEquations2D{NVARS, NCOMP, RealT}(__gammas,
                                                                  __gas_constants,
                                                                  __c_p, __c_v, L_00)
end

@inline function Base.real(::CompressibleMoistEulerEquations2D{NVARS, NCOMP,
                                                               RealT}) where {NVARS,
                                                                              NCOMP,
                                                                              RealT}
    RealT
end

"""
    totalgamma(u, equations::CompressibleMoistEulerEquations2D)

Function that calculates the total gamma out of all partial gammas using the
partial density fractions as well as the partial specific heats at constant volume.
"""
@inline function totalgamma(u, equations::CompressibleMoistEulerEquations2D)
    @unpack c_v, c_p = equations

    help1 = zero(u[1])
    help2 = zero(u[1])

    # compute weighted averages of cp and cv
    # normalization by total rho not required, would cancel below
    for i in eachcomponent(equations)
        help1 += u[i + 3] * c_p[i]
        help2 += u[i + 3] * c_v[i]
    end

    return help1 / help2
end

"""
pressure(u, equations::CompressibleMoistEulerEquations2D)

Calculate pressure. This differs from the calculation in `AbstractCompressibleEulerMulticomponentEquations` in that the latent heat is accounted for.
"""
@inline function pressure(u, equations::CompressibleMoistEulerEquations2D)
    @unpack L_00 = equations
    rho_v1, rho_v2, rho_e, rho_d, rho_v = u
    rho = density(u, equations)
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    gamma = totalgamma(u, equations)
    p = (gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2) - L_00 * rho_v)
    return p
end

"""
temperature(u, equations::CompressibleMoistEulerEquations2D)

Calculate temperature. Account for latent heat.
"""
@inline function temperature(u, equations::CompressibleMoistEulerEquations2D)
    @unpack c_v, gammas, gas_constants, L_00 = equations
    rho_v1, rho_v2, rho_e, rho_d, rho_v = u

    rho = density(u, equations)
    help1 = zero(rho)

    # compute weighted average of cv
    # normalization by rho not required, cancels below
    for i in eachcomponent(equations)
        help1 += u[i + 3] * c_v[i]
    end

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho

    T = (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2) - L_00 * rho_v) / help1
    return T
end

"""
    density_gas_constant(u, equations::CompressibleMoistEulerEquations2D)

Function that calculates overall density times overall gas constant.
"""
@inline function density_gas_constant(u,
                                      equations::CompressibleMoistEulerEquations2D)
    @unpack gas_constants = equations
    help = zero(u[1])
    for i in eachcomponent(equations)
        help += u[i + 3] * gas_constant[i]
    end
    return help
end


"""
cons2entropy(u, equations::CompressibleMoistEulerEquations2D)

Convert conservative variables to entropy.
"""
@inline function cons2entropy(u, equations::CompressibleMoistEulerEquations2D)
    @unpack c_v, gammas, gas_constants = equations
    rho_v1, rho_v2, rho_e = u

    rho = density(u, equations)

    # Multicomponent stuff
    help1 = zero(rho)
    gas_constant = zero(rho)
    for i in eachcomponent(equations)
        help1 += u[i + 3] * c_v[i]
        gas_constant += gas_constants[i] * (u[i + 3] / rho)
    end

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_square = v1^2 + v2^2
    p = pressure(u, equations)
    rho_p = rho / p
    T = temperature(u, equations)

    entrop_rho = SVector{ncomponents(equations), real(equations)}((c_v[i] *
                                                                   (1 - log(T)) +
                                                                   gas_constants[i] *
                                                                   (1 + log(u[i + 3])) -
                                                                   v_square / (2 * T))
                                                                  for i in eachcomponent(equations))

    w1 = gas_constant * v1 * rho_p
    w2 = gas_constant * v2 * rho_p
    w3 = gas_constant * (-rho_p)

    entrop_other = SVector{3, real(equations)}(w1, w2, w3)

    return vcat(entrop_other, entrop_rho)
end

"""
entropy2cons(w, equations::CompressibleMoistEulerEquations2D)

Convert entropy variables to conservative variables
"""
@inline function entropy2cons(w, equations::CompressibleMoistEulerEquations2D)
    @unpack gammas, gas_constants, cp, cv = equations
    T = -1 / w[3]
    v1 = w[1] * T
    v2 = w[2] * T
    v_squared = v1^2 + v2^2
    cons_rho = SVector{ncomponents(equations), real(equations)}(exp((w[i + 3] -
                                                                     cv[i] *
                                                                     (1 - log(T)) +
                                                                     v_squared /
                                                                     (2 * T)) /
                                                                    gas_constants[i] -
                                                                    1)
                                                                for i in eachcomponent(equations))

    rho = zero(cons_rho[1])
    help1 = zero(cons_rho[1])
    help2 = zero(cons_rho[1])
    p = zero(cons_rho[1])
    for i in eachcomponent(equations)
        rho += cons_rho[i]
        help1 += cons_rho[i] * cv[i] * gammas[i]
        help2 += cons_rho[i] * cv[i]
        p += cons_rho[i] * gas_constants[i] * T
    end
    u1 = rho * v1
    u2 = rho * v2
    gamma = help1 / help2
    u3 = p / (gamma - 1) + 0.5 * rho * v_squared + L_00 * cons_rho[2]
    cons_other = SVector{3, real(equations)}(u1, u2, u3)
    return vcat(cons_other, cons_rho)
end

"""
    flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleMoistEulerEquations2D)

Adaption of the entropy conserving two-point flux by
- Ayoub Gouasmi, Karthik Duraisamy (2020)
  "Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations"
  [arXiv:1904.00972v3](https://arxiv.org/abs/1904.00972) [math.NA] 4 Feb 2020
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation::Integer,
                                    equations::CompressibleMoistEulerEquations2D)
    # Unpack left and right state
    @unpack gammas, gas_constants, cv, L_00 = equations
    rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
    rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
    rhok_mean = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i + 3],
                                                                         u_rr[i + 3])
                                                                 for i in eachcomponent(equations))
    rhok_avg = SVector{ncomponents(equations), real(equations)}(0.5 * (u_ll[i + 3] +
                                                                 u_rr[i + 3])
                                                                for i in eachcomponent(equations))

    # Iterating over all partial densities
    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)

    # extract velocities
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    v1_avg = 0.5 * (v1_ll + v1_rr)
    v2_avg = 0.5 * (v2_ll + v2_rr)
    v1_square = 0.5 * (v1_ll^2 + v1_rr^2)
    v2_square = 0.5 * (v2_ll^2 + v2_rr^2)
    v_sum = v1_avg + v2_avg

    enth = zero(v_sum)
    for i in eachcomponent(equations)
        enth += rhok_avg[i] * gas_constants[i]
    end

    T_ll = temperature(u_ll, equations)
    T_rr = temperature(u_rr, equations)
    T = 0.5 * (1.0 / T_ll + 1.0 / T_rr)
    T_log = ln_mean(1.0 / T_ll, 1.0 / T_rr)

    # Calculate fluxes depending on orientation
    help1 = zero(T_ll)
    help2 = zero(T_rr)
    if orientation == 1
        f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v1_avg
                                                                 for i in eachcomponent(equations))
        for i in eachcomponent(equations)
            help1 += f_rho[i] * cv[i]
            help2 += f_rho[i]
        end
        f1 = (help2) * v1_avg + enth / T
        f2 = (help2) * v2_avg
        # Account for latent heat. This is the only difference compared to 
        # AbstractCompressibleEulerMulticomponentEquations
        f3 = (help1) / T_log - 0.5 * (v1_square + v2_square) * (help2) + v1_avg * f1 +
             v2_avg * f2 + L_00 * f_rho[2]
    else
        f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v2_avg
                                                                 for i in eachcomponent(equations))
        for i in eachcomponent(equations)
            help1 += f_rho[i] * cv[i]
            help2 += f_rho[i]
        end
        f1 = (help2) * v1_avg
        f2 = (help2) * v2_avg + enth / T
        # Account for latent heat. This is the only difference compared to 
        # AbstractCompressibleEulerMulticomponentEquations
        f3 = (help1) / T_log - 0.5 * (v1_square + v2_square) * (help2) + v1_avg * f1 +
             v2_avg * f2 + L_00 * f_rho[2]
    end
    f_other = SVector{3, real(equations)}(f1, f2, f3)

    return vcat(f_other, f_rho)
end

@inline function flux_chandrashekar(u_ll, u_rr, normal_direction::AbstractVector,
                                    equations::CompressibleMoistEulerEquations2D)
    # Unpack left and right state
    @unpack gammas, gas_constants, c_v, L_00 = equations
    rho_v1_ll, rho_v2_ll = u_ll
    rho_v1_rr, rho_v2_rr = u_rr
    rhok_mean = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i + 3],
                                                                         u_rr[i + 3])
                                                                 for i in eachcomponent(equations))
    rhok_avg = SVector{ncomponents(equations), real(equations)}(0.5 * (u_ll[i + 3] +
                                                                 u_rr[i + 3])
                                                                for i in eachcomponent(equations))

    # Iterating over all partial densities
    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)

    # extract velocities
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    v1_avg = 0.5 * (v1_ll + v1_rr)
    v2_avg = 0.5 * (v2_ll + v2_rr)
    v_dot_n_avg = normal_direction[1] * v1_avg + normal_direction[2] * v2_avg
    v1_square = 0.5 * (v1_ll^2 + v1_rr^2)
    v2_square = 0.5 * (v2_ll^2 + v2_rr^2)
    v_sum = v1_avg + v2_avg

    enth = zero(v_sum)
    for i in eachcomponent(equations)
        enth += rhok_avg[i] * gas_constants[i]
    end

    T_ll = temperature(u_ll, equations)
    T_rr = temperature(u_rr, equations)
    T = 0.5 * (1.0 / T_ll + 1.0 / T_rr)
    T_log = ln_mean(1.0 / T_ll, 1.0 / T_rr)

    f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v_dot_n_avg
                                                             for i in eachcomponent(equations))

    help1 = zero(T_ll)
    help2 = zero(T_rr)
    for i in eachcomponent(equations)
        help1 += f_rho[i] * c_v[i]
        help2 += f_rho[i]
    end

    f1 = (help2) * v1_avg + normal_direction[1] * enth / T
    f2 = (help2) * v2_avg + normal_direction[2] * enth / T

    # Account for latent heat. This is the only difference compared to 
    # AbstractCompressibleEulerMulticomponentEquations
    f3 = ((help1) / T_log - 0.5 * (v1_square + v2_square) * (help2)
          + v1_avg * f1 + v2_avg * f2 + L_00 * f_rho[2])

    f_other = SVector{3, real(equations)}(f1, f2, f3)

    return vcat(f_other, f_rho)
end
end # @muladd
