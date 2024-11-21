# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    CompressibleEulerMulticomponentEquations1D(; gammas, gas_constants)

Multicomponent version of the compressible Euler equations
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
\rho v_1 \\ \rho e \\ \rho_1 \\ \rho_2 \\ \vdots \\ \rho_{n}
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
\rho v_1^2 + p \\ (\rho e +p) v_1 \\ \rho_1 v_1 \\ \rho_2 v_1 \\ \vdots \\ \rho_{n} v_1
\end{pmatrix}

=
\begin{pmatrix}
0 \\ 0 \\ 0 \\ 0 \\ \vdots \\ 0
\end{pmatrix}
```
for calorically perfect gas in one space dimension.
Here, ``\rho_i`` is the density of component ``i``, ``\rho=\sum_{i=1}^n\rho_i`` the sum of the individual ``\rho_i``,
``v_1`` the velocity, ``e`` the specific total energy **rather than** specific internal energy, and
```math
p = (\gamma - 1) \left( \rho e - \frac{1}{2} \rho v_1^2 \right)
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
`gas_constants` should be passed as tuples, e.g., `gammas=(1.4, 1.667)`.

The remaining variables like the specific heats at constant volume `cv` or the specific heats at
constant pressure `cp` are then calculated considering a calorically perfect gas.
"""
struct CompressibleEulerMulticomponentEquations1D{NVARS, NCOMP, RealT <: Real} <:
       AbstractCompressibleEulerMulticomponentEquations{1, NVARS, NCOMP}
    gammas::SVector{NCOMP, RealT}
    gas_constants::SVector{NCOMP, RealT}
    cv::SVector{NCOMP, RealT}
    cp::SVector{NCOMP, RealT}

    function CompressibleEulerMulticomponentEquations1D{NVARS, NCOMP, RealT}(gammas::SVector{NCOMP,
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

function CompressibleEulerMulticomponentEquations1D(; gammas, gas_constants)
    _gammas = promote(gammas...)
    _gas_constants = promote(gas_constants...)
    RealT = promote_type(eltype(_gammas), eltype(_gas_constants),
                         typeof(gas_constants[1] / (gammas[1] - 1)))

    NVARS = length(_gammas) + 2
    NCOMP = length(_gammas)

    __gammas = SVector(map(RealT, _gammas))
    __gas_constants = SVector(map(RealT, _gas_constants))

    return CompressibleEulerMulticomponentEquations1D{NVARS, NCOMP, RealT}(__gammas,
                                                                           __gas_constants)
end

@inline function Base.real(::CompressibleEulerMulticomponentEquations1D{NVARS, NCOMP,
                                                                        RealT}) where {
                                                                                       NVARS,
                                                                                       NCOMP,
                                                                                       RealT
                                                                                       }
    RealT
end

function varnames(::typeof(cons2cons),
                  equations::CompressibleEulerMulticomponentEquations1D)
    cons = ("rho_v1", "rho_e")
    rhos = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
    return (cons..., rhos...)
end

function varnames(::typeof(cons2prim),
                  equations::CompressibleEulerMulticomponentEquations1D)
    prim = ("v1", "p")
    rhos = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
    return (prim..., rhos...)
end

# Set initial conditions at physical location `x` for time `t`

"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerMulticomponentEquations1D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t,
                                            equations::CompressibleEulerMulticomponentEquations1D)
    RealT = eltype(x)
    c = 2
    A = convert(RealT, 0.1)
    L = 2
    f = 1.0f0 / L
    omega = 2 * convert(RealT, pi) * f
    ini = c + A * sin(omega * (x[1] - t))

    v1 = 1

    rho = ini

    # Here we compute an arbitrary number of different rhos. (one rho is double the next rho while the sum of all rhos is 1)
    prim_rho = SVector{ncomponents(equations), real(equations)}(2^(i - 1) * (1 - 2) *
                                                                rho / (1 -
                                                                 2^ncomponents(equations))
                                                                for i in eachcomponent(equations))

    prim1 = rho * v1
    prim2 = rho^2

    prim_other = SVector(prim1, prim2)

    return vcat(prim_other, prim_rho)
end

"""
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerMulticomponentEquations1D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t,
                                               equations::CompressibleEulerMulticomponentEquations1D)
    # Same settings as in `initial_condition`
    RealT = eltype(u)
    c = 2
    A = convert(RealT, 0.1)
    L = 2
    f = 1.0f0 / L
    omega = 2 * convert(RealT, pi) * f

    gamma = totalgamma(u, equations)

    x1, = x
    si, co = sincos((t - x1) * omega)
    tmp = (-((4 * si * A - 4 * c) + 1) * (gamma - 1) * co * A * omega) / 2

    # Here we compute an arbitrary number of different rhos. (one rho is double the next rho while the sum of all rhos is 1
    du_rho = SVector{ncomponents(equations), real(equations)}(0
                                                              for i in eachcomponent(equations))

    du1 = tmp
    du2 = tmp

    du_other = SVector(du1, du2)

    return vcat(du_other, du_rho)
end

"""
    initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerMulticomponentEquations1D)

A for multicomponent adapted weak blast wave adapted to multicomponent and taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t,
                                           equations::CompressibleEulerMulticomponentEquations1D)
    # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
    RealT = eltype(x)
    inicenter = SVector(0)
    x_norm = x[1] - inicenter[1]
    r = abs(x_norm)
    cos_phi = x_norm > 0 ? 1 : -1

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
    p = r > 0.5f0 ? one(RealT) : convert(RealT, 1.245)

    prim_other = SVector(v1, p)

    return prim2cons(vcat(prim_other, prim_rho), equations)
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer,
                      equations::CompressibleEulerMulticomponentEquations1D)
    rho_v1, rho_e = u

    rho = density(u, equations)

    v1 = rho_v1 / rho
    gamma = totalgamma(u, equations)
    p = (gamma - 1) * (rho_e - 0.5f0 * rho * v1^2)

    f_rho = densities(u, v1, equations)
    f1 = rho_v1 * v1 + p
    f2 = (rho_e + p) * v1

    f_other = SVector(f1, f2)

    return vcat(f_other, f_rho)
end

"""
    flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations1D)

Entropy conserving two-point flux by
- Ayoub Gouasmi, Karthik Duraisamy (2020)
  "Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations"
  [arXiv:1904.00972v3](https://arxiv.org/abs/1904.00972) [math.NA] 4 Feb 2020
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation::Integer,
                                    equations::CompressibleEulerMulticomponentEquations1D)
    # Unpack left and right state
    @unpack gammas, gas_constants, cv = equations
    rho_v1_ll, rho_e_ll = u_ll
    rho_v1_rr, rho_e_rr = u_rr
    rhok_mean = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i + 2],
                                                                         u_rr[i + 2])
                                                                 for i in eachcomponent(equations))
    rhok_avg = SVector{ncomponents(equations), real(equations)}(0.5f0 * (u_ll[i + 2] +
                                                                 u_rr[i + 2])
                                                                for i in eachcomponent(equations))

    # Iterating over all partial densities
    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)

    gamma_ll = totalgamma(u_ll, equations)
    gamma_rr = totalgamma(u_rr, equations)

    # extract velocities
    v1_ll = rho_v1_ll / rho_ll
    v1_rr = rho_v1_rr / rho_rr
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v1_square = 0.5f0 * (v1_ll^2 + v1_rr^2)
    v_sum = v1_avg

    RealT = eltype(u_ll)
    enth = zero(RealT)
    help1_ll = zero(RealT)
    help1_rr = zero(RealT)

    for i in eachcomponent(equations)
        enth += rhok_avg[i] * gas_constants[i]
        help1_ll += u_ll[i + 2] * cv[i]
        help1_rr += u_rr[i + 2] * cv[i]
    end

    T_ll = (rho_e_ll - 0.5f0 * rho_ll * (v1_ll^2)) / help1_ll
    T_rr = (rho_e_rr - 0.5f0 * rho_rr * (v1_rr^2)) / help1_rr
    T = 0.5f0 * (1 / T_ll + 1 / T_rr)
    T_log = ln_mean(1 / T_ll, 1 / T_rr)

    # Calculate fluxes depending on orientation
    help1 = zero(RealT)
    help2 = zero(RealT)

    f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v1_avg
                                                             for i in eachcomponent(equations))
    for i in eachcomponent(equations)
        help1 += f_rho[i] * cv[i]
        help2 += f_rho[i]
    end
    f1 = (help2) * v1_avg + enth / T
    f2 = (help1) / T_log - 0.5f0 * (v1_square) * (help2) + v1_avg * f1

    f_other = SVector(f1, f2)

    return vcat(f_other, f_rho)
end

"""
    flux_ranocha(u_ll, u_rr, orientation_or_normal_direction,
                 equations::CompressibleEulerMulticomponentEquations1D)

Adaption of the entropy conserving and kinetic energy preserving two-point flux by
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
@inline function flux_ranocha(u_ll, u_rr, orientation::Integer,
                              equations::CompressibleEulerMulticomponentEquations1D)
    # Unpack left and right state
    @unpack gammas, gas_constants, cv = equations
    rho_v1_ll, rho_e_ll = u_ll
    rho_v1_rr, rho_e_rr = u_rr
    rhok_mean = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i + 2],
                                                                         u_rr[i + 2])
                                                                 for i in eachcomponent(equations))
    rhok_avg = SVector{ncomponents(equations), real(equations)}(0.5f0 * (u_ll[i + 2] +
                                                                 u_rr[i + 2])
                                                                for i in eachcomponent(equations))

    # Iterating over all partial densities
    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)

    # Calculating gamma
    gamma = totalgamma(0.5f0 * (u_ll + u_rr), equations)
    inv_gamma_minus_one = 1 / (gamma - 1)

    # extract velocities
    v1_ll = rho_v1_ll / rho_ll
    v1_rr = rho_v1_rr / rho_rr
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    velocity_square_avg = 0.5f0 * (v1_ll * v1_rr)

    # density flux
    f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v1_avg
                                                             for i in eachcomponent(equations))

    # helpful variables
    RealT = eltype(u_ll)
    f_rho_sum = zero(RealT)
    help1_ll = zero(RealT)
    help1_rr = zero(RealT)
    enth_ll = zero(RealT)
    enth_rr = zero(RealT)
    for i in eachcomponent(equations)
        enth_ll += u_ll[i + 2] * gas_constants[i]
        enth_rr += u_rr[i + 2] * gas_constants[i]
        f_rho_sum += f_rho[i]
        help1_ll += u_ll[i + 2] * cv[i]
        help1_rr += u_rr[i + 2] * cv[i]
    end

    # temperature and pressure
    T_ll = (rho_e_ll - 0.5f0 * rho_ll * (v1_ll^2)) / help1_ll
    T_rr = (rho_e_rr - 0.5f0 * rho_rr * (v1_rr^2)) / help1_rr
    p_ll = T_ll * enth_ll
    p_rr = T_rr * enth_rr
    p_avg = 0.5f0 * (p_ll + p_rr)
    inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)

    # momentum and energy flux
    f1 = f_rho_sum * v1_avg + p_avg
    f2 = f_rho_sum * (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one) +
         0.5f0 * (p_ll * v1_rr + p_rr * v1_ll)
    f_other = SVector(f1, f2)

    return vcat(f_other, f_rho)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::CompressibleEulerMulticomponentEquations1D)
    rho_v1_ll, rho_e_ll = u_ll
    rho_v1_rr, rho_e_rr = u_rr

    # Calculate primitive variables and speed of sound
    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)
    gamma_ll = totalgamma(u_ll, equations)
    gamma_rr = totalgamma(u_rr, equations)

    v_ll = rho_v1_ll / rho_ll
    v_rr = rho_v1_rr / rho_rr

    p_ll = (gamma_ll - 1) * (rho_e_ll - 0.5f0 * rho_ll * v_ll^2)
    p_rr = (gamma_rr - 1) * (rho_e_rr - 0.5f0 * rho_rr * v_rr^2)
    c_ll = sqrt(gamma_ll * p_ll / rho_ll)
    c_rr = sqrt(gamma_rr * p_rr / rho_rr)

    λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end

@inline function max_abs_speeds(u,
                                equations::CompressibleEulerMulticomponentEquations1D)
    rho_v1, rho_e = u

    rho = density(u, equations)
    v1 = rho_v1 / rho

    gamma = totalgamma(u, equations)
    p = (gamma - 1) * (rho_e - 0.5f0 * rho * (v1^2))
    c = sqrt(gamma * p / rho)

    return (abs(v1) + c,)
end

# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerMulticomponentEquations1D)
    rho_v1, rho_e = u

    prim_rho = SVector{ncomponents(equations), real(equations)}(u[i + 2]
                                                                for i in eachcomponent(equations))

    rho = density(u, equations)
    v1 = rho_v1 / rho
    gamma = totalgamma(u, equations)

    p = (gamma - 1) * (rho_e - 0.5f0 * rho * (v1^2))
    prim_other = SVector(v1, p)

    return vcat(prim_other, prim_rho)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerMulticomponentEquations1D)
    @unpack cv, gammas = equations
    v1, p = prim

    RealT = eltype(prim)

    cons_rho = SVector{ncomponents(equations), RealT}(prim[i + 2]
                                                      for i in eachcomponent(equations))
    rho = density(prim, equations)
    gamma = totalgamma(prim, equations)

    rho_v1 = rho * v1

    rho_e = p / (gamma - 1) + 0.5f0 * (rho_v1 * v1)

    cons_other = SVector{2, RealT}(rho_v1, rho_e)

    return vcat(cons_other, cons_rho)
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerMulticomponentEquations1D)
    @unpack cv, gammas, gas_constants = equations

    rho_v1, rho_e = u

    rho = density(u, equations)

    RealT = eltype(u)
    help1 = zero(RealT)
    gas_constant = zero(RealT)
    for i in eachcomponent(equations)
        help1 += u[i + 2] * cv[i]
        gas_constant += gas_constants[i] * (u[i + 2] / rho)
    end

    v1 = rho_v1 / rho
    v_square = v1^2
    gamma = totalgamma(u, equations)

    p = (gamma - 1) * (rho_e - 0.5f0 * rho * v_square)
    s = log(p) - gamma * log(rho) - log(gas_constant)
    rho_p = rho / p
    T = (rho_e - 0.5f0 * rho * v_square) / (help1)

    entrop_rho = SVector{ncomponents(equations), real(equations)}((cv[i] *
                                                                   (1 - log(T)) +
                                                                   gas_constants[i] *
                                                                   (1 + log(u[i + 2])) -
                                                                   v1^2 / (2 * T))
                                                                  for i in eachcomponent(equations))

    w1 = gas_constant * v1 * rho_p
    w2 = gas_constant * (-rho_p)

    entrop_other = SVector(w1, w2)

    return vcat(entrop_other, entrop_rho)
end

# Convert entropy variables to conservative variables
@inline function entropy2cons(w, equations::CompressibleEulerMulticomponentEquations1D)
    @unpack gammas, gas_constants, cv, cp = equations
    T = -1 / w[2]
    v1 = w[1] * T
    cons_rho = SVector{ncomponents(equations), real(equations)}(exp(1 /
                                                                    gas_constants[i] *
                                                                    (-cv[i] *
                                                                     log(-w[2]) -
                                                                     cp[i] + w[i + 2] -
                                                                     0.5f0 * w[1]^2 /
                                                                     w[2]))
                                                                for i in eachcomponent(equations))

    RealT = eltype(w)
    rho = zero(RealT)
    help1 = zero(RealT)
    help2 = zero(RealT)
    p = zero(RealT)
    for i in eachcomponent(equations)
        rho += cons_rho[i]
        help1 += cons_rho[i] * cv[i] * gammas[i]
        help2 += cons_rho[i] * cv[i]
        p += cons_rho[i] * gas_constants[i] * T
    end
    u1 = rho * v1
    gamma = help1 / help2
    u2 = p / (gamma - 1) + 0.5f0 * rho * v1^2
    cons_other = SVector(u1, u2)
    return vcat(cons_other, cons_rho)
end

@inline function total_entropy(u, equations::CompressibleEulerMulticomponentEquations1D)
    @unpack cv, gammas, gas_constants = equations
    rho_v1, rho_e = u
    rho = density(u, equations)
    T = temperature(u, equations)

    RealT = eltype(u)
    total_entropy = zero(RealT)
    for i in eachcomponent(equations)
        total_entropy -= u[i + 2] * (cv[i] * log(T) - gas_constants[i] * log(u[i + 2]))
    end

    return total_entropy
end

@inline function temperature(u, equations::CompressibleEulerMulticomponentEquations1D)
    @unpack cv, gammas, gas_constants = equations

    rho_v1, rho_e = u

    rho = density(u, equations)

    RealT = eltype(u)
    help1 = zero(RealT)

    for i in eachcomponent(equations)
        help1 += u[i + 2] * cv[i]
    end

    v1 = rho_v1 / rho
    v_square = v1^2
    T = (rho_e - 0.5f0 * rho * v_square) / help1

    return T
end

"""
    totalgamma(u, equations::CompressibleEulerMulticomponentEquations1D)

Function that calculates the total gamma out of all partial gammas using the
partial density fractions as well as the partial specific heats at constant volume.
"""
@inline function totalgamma(u, equations::CompressibleEulerMulticomponentEquations1D)
    @unpack cv, gammas = equations

    RealT = eltype(u)
    help1 = zero(RealT)
    help2 = zero(RealT)

    for i in eachcomponent(equations)
        help1 += u[i + 2] * cv[i] * gammas[i]
        help2 += u[i + 2] * cv[i]
    end

    return help1 / help2
end

@inline function pressure(u, equations::CompressibleEulerMulticomponentEquations1D)
    rho_v1, rho_e = u

    rho = density(u, equations)
    gamma = totalgamma(u, equations)

    p = (gamma - 1) * (rho_e - 0.5f0 * (rho_v1^2) / rho)

    return p
end

@inline function density(u, equations::CompressibleEulerMulticomponentEquations1D)
    RealT = eltype(u)
    rho = zero(RealT)

    for i in eachcomponent(equations)
        rho += u[i + 2]
    end

    return rho
end

@inline function densities(u, v, equations::CompressibleEulerMulticomponentEquations1D)
    return SVector{ncomponents(equations), real(equations)}(u[i + 2] * v
                                                            for i in eachcomponent(equations))
end

@inline function velocity(u, equations::CompressibleEulerMulticomponentEquations1D)
    rho = density(u, equations)
    v1 = u[1] / rho
    return v1
end
end # @muladd
