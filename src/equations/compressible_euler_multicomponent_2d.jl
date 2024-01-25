# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    CompressibleEulerMulticomponentEquations2D(; gammas, gas_constants)

Multicomponent version of the compressible Euler equations
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
\rho v_1 \\ \rho v_2 \\ \rho e \\ \rho_1 \\ \rho_2 \\ \vdots \\ \rho_{n}
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
\rho v_1^2 + p \\ \rho v_1 v_2 \\ ( \rho e +p) v_1 \\ \rho_1 v_1 \\ \rho_2 v_1 \\ \vdots \\ \rho_{n} v_1
\end{pmatrix}
+
\frac{\partial}{\partial y}
\begin{pmatrix}
\rho v_1 v_2 \\ \rho v_2^2 + p \\ ( \rho e +p) v_2 \\ \rho_1 v_2 \\ \rho_2 v_2 \\ \vdots \\ \rho_{n} v_2
\end{pmatrix}
=
\begin{pmatrix}
0 \\ 0 \\ 0 \\ 0 \\ 0 \\ \vdots \\ 0
\end{pmatrix}
```
for calorically perfect gas in two space dimensions.
Here, ``\rho_i`` is the density of component ``i``, ``\rho=\sum_{i=1}^n\rho_i`` the sum of the individual ``\rho_i``,
``v_1``, ``v_2`` the velocities, ``e`` the specific total energy **rather than** specific internal energy, and
```math
p = (\gamma - 1) \left( \rho e - \frac{1}{2} \rho (v_1^2 + v_2^2) \right)
```
the pressure,
```math
\gamma=\frac{\sum_{i=1}^n\rho_i C_{v,i}\gamma_i}{\sum_{i=1}^n\rho_i C_{v,i}}
```
total heat capacity ratio, ``\gamma_i`` heat capacity ratio of component ``i``,
```math
C_{v,i}=\frac{R}{\gamma_i-1}
```
specific heat capacity at constant volume of component ``i``.

In case of more than one component, the specific heat ratios `gammas` and the gas constants
`gas_constants` in [kJ/(kg*K)] should be passed as tuples, e.g., `gammas=(1.4, 1.667)`.

The remaining variables like the specific heats at constant volume `cv` or the specific heats at
constant pressure `cp` are then calculated considering a calorically perfect gas.
"""
struct CompressibleEulerMulticomponentEquations2D{NVARS, NCOMP, RealT <: Real} <:
       AbstractCompressibleEulerMulticomponentEquations{2, NVARS, NCOMP}
    gammas::SVector{NCOMP, RealT}
    gas_constants::SVector{NCOMP, RealT}
    cv::SVector{NCOMP, RealT}
    cp::SVector{NCOMP, RealT}

    function CompressibleEulerMulticomponentEquations2D{NVARS, NCOMP, RealT}(gammas::SVector{NCOMP,
                                                                                             RealT},
                                                                             gas_constants::SVector{NCOMP,
                                                                                                    RealT}) where {
                                                                                                                   NVARS,
                                                                                                                   NCOMP,
                                                                                                                   RealT <:
                                                                                                                   Real
                                                                                                                   }
        NCOMP >= 1 ||
            throw(DimensionMismatch("`gammas` and `gas_constants` have to be filled with at least one value"))

        cv = gas_constants ./ (gammas .- 1)
        cp = gas_constants + gas_constants ./ (gammas .- 1)

        new(gammas, gas_constants, cv, cp)
    end
end

function CompressibleEulerMulticomponentEquations2D(; gammas, gas_constants)
    _gammas = promote(gammas...)
    _gas_constants = promote(gas_constants...)
    RealT = promote_type(eltype(_gammas), eltype(_gas_constants),
                         typeof(gas_constants[1] / (gammas[1] - 1)))

    NVARS = length(_gammas) + 3
    NCOMP = length(_gammas)

    __gammas = SVector(map(RealT, _gammas))
    __gas_constants = SVector(map(RealT, _gas_constants))

    return CompressibleEulerMulticomponentEquations2D{NVARS, NCOMP, RealT}(__gammas,
                                                                           __gas_constants)
end

@inline function Base.real(::CompressibleEulerMulticomponentEquations2D{NVARS, NCOMP,
                                                                        RealT}) where {
                                                                                       NVARS,
                                                                                       NCOMP,
                                                                                       RealT
                                                                                       }
    RealT
end

"""
    temperature(u, equations::CompressibleEulerMulticomponentEquations2D)

Calculate temperature.
"""
@inline function temperature(u, equations::CompressibleEulerMulticomponentEquations2D)
    rho_v1, rho_v2, rho_e = u
    @unpack cv = equations

    rho = density(u, equations)
    help1 = zero(rho)

    # compute weighted average of cv
    # normalization by rho not required, cancels below
    for i in eachcomponent(equations)
        help1 += u[i + 3] * cv[i]
    end

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_square = v1^2 + v2^2
    T = (rho_e - 0.5 * rho * v_square) / help1
    return T
end

"""
    totalgamma(u, equations::CompressibleEulerMulticomponentEquations2D)

Function that calculates the total gamma out of all partial gammas using the
partial density fractions as well as the partial specific heats at constant volume.
"""
@inline function totalgamma(u, equations::CompressibleEulerMulticomponentEquations2D)
    @unpack cv, cp = equations

    help1 = zero(u[1])
    help2 = zero(u[1])

    # compute weighted averages of cp and cv
    # normalization by total rho not required, would cancel below
    for i in eachcomponent(equations)
        help1 += u[i + 3] * cp[i]
        help2 += u[i + 3] * cv[i]
    end

    return help1 / help2
end

"""
cons2entropy(u, equations::CompressibleEulerMulticomponentEquations2D)

Convert conservative variables to entropy.
"""
@inline function cons2entropy(u, equations::CompressibleEulerMulticomponentEquations2D)
    @unpack cv, gammas, gas_constants = equations
    rho_v1, rho_v2, rho_e = u

    rho = density(u, equations)

    # Multicomponent stuff
    help1 = zero(rho)
    gas_constant = zero(rho)
    for i in eachcomponent(equations)
        help1 += u[i + 3] * cv[i]
        gas_constant += gas_constants[i] * (u[i + 3] / rho)
    end

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_square = v1^2 + v2^2
    p = pressure(u, equations)
    rho_p = rho / p
    T = temperature(u, equations)

    entrop_rho = SVector{ncomponents(equations), real(equations)}((cv[i] *
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
entropy2cons(w, equations::CompressibleEulerMulticomponentEquations2D)

Convert entropy variables to conservative variables
"""
@inline function entropy2cons(w, equations::CompressibleEulerMulticomponentEquations2D)
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
    u3 = p / (gamma - 1) + 0.5 * rho * v_squared
    cons_other = SVector{3, real(equations)}(u1, u2, u3)
    return vcat(cons_other, cons_rho)
end

"""
    total_entropy(u, equations::CompressibleEulerMulticomponentEquations2D)

Calculate total entropy.
"""
@inline function total_entropy(u, equations::CompressibleEulerMulticomponentEquations2D)
    @unpack cv, gammas, gas_constants = equations
    T = temperature(u, equations)

    total_entropy = zero(u[1])
    for i in eachcomponent(equations)
        total_entropy -= u[i + 3] * (cv[i] * log(T) - gas_constants[i] * log(u[i + 3]))
    end

    return total_entropy
end

"""
    density_gas_constant(u, equations::CompressibleEulerMulticomponentEquations2D{2})

Function that calculates overall density times overall gas constant.
"""
@inline function density_gas_constant(u,
                                      equations::CompressibleEulerMulticomponentEquations2D)
    @unpack gas_constants = equations
    help = zero(u[1])
    for i in eachcomponent(equations)
        help += u[i + 3] * gas_constants[i]
    end
    return help
end

"""
    flux_chandrashekar(u_ll, u_rr, orientation_or_normal_direction, equations::CompressibleEulerMulticomponentEquations2D)

Adaption of the entropy conserving two-point flux by
- Ayoub Gouasmi, Karthik Duraisamy (2020)
  "Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations"
  [arXiv:1904.00972v3](https://arxiv.org/abs/1904.00972) [math.NA] 4 Feb 2020
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation::Integer,
                                    equations::CompressibleEulerMulticomponentEquations2D)
    # Unpack left and right state
    @unpack gammas, gas_constants, cv = equations
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
        f3 = (help1) / T_log - 0.5 * (v1_square + v2_square) * (help2) + v1_avg * f1 +
             v2_avg * f2
    else
        f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v2_avg
                                                                 for i in eachcomponent(equations))
        for i in eachcomponent(equations)
            help1 += f_rho[i] * cv[i]
            help2 += f_rho[i]
        end
        f1 = (help2) * v1_avg
        f2 = (help2) * v2_avg + enth / T
        f3 = (help1) / T_log - 0.5 * (v1_square + v2_square) * (help2) + v1_avg * f1 +
             v2_avg * f2
    end
    f_other = SVector{3, real(equations)}(f1, f2, f3)

    return vcat(f_other, f_rho)
end

@inline function flux_chandrashekar(u_ll, u_rr, normal_direction::AbstractVector,
                                    equations::CompressibleEulerMulticomponentEquations2D)
    # Unpack left and right state
    @unpack gammas, gas_constants, cv = equations
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
        help1 += f_rho[i] * cv[i]
        help2 += f_rho[i]
    end

    f1 = (help2) * v1_avg + normal_direction[1] * enth / T
    f2 = (help2) * v2_avg + normal_direction[2] * enth / T
    f3 = ((help1) / T_log - 0.5 * (v1_square + v2_square) * (help2)
          + v1_avg * f1 + v2_avg * f2)

    f_other = SVector{3, real(equations)}(f1, f2, f3)

    return vcat(f_other, f_rho)
end
end # @muladd
