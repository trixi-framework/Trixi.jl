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

function varnames(::typeof(cons2cons),
                  equations::CompressibleEulerMulticomponentEquations2D)
    cons = ("rho_v1", "rho_v2", "rho_e")
    rhos = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
    return (cons..., rhos...)
end

function varnames(::typeof(cons2prim),
                  equations::CompressibleEulerMulticomponentEquations2D)
    prim = ("v1", "v2", "p")
    rhos = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
    return (prim..., rhos...)
end

# Set initial conditions at physical location `x` for time `t`

"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerMulticomponentEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t,
                                            equations::CompressibleEulerMulticomponentEquations2D)
    RealT = eltype(x)
    c = 2
    A = convert(RealT, 0.1)
    L = 2
    f = 1.0f0 / L
    omega = 2 * convert(RealT, pi) * f
    ini = c + A * sin(omega * (x[1] + x[2] - t))

    v1 = 1
    v2 = 1

    rho = ini

    # Here we compute an arbitrary number of different rhos. (one rho is double the next rho while the sum of all rhos is 1)
    prim_rho = SVector{ncomponents(equations), real(equations)}(2^(i - 1) * (1 - 2) *
                                                                rho / (1 -
                                                                 2^ncomponents(equations))
                                                                for i in eachcomponent(equations))

    prim1 = rho * v1
    prim2 = rho * v2
    prim3 = rho^2

    prim_other = SVector(prim1, prim2, prim3)

    return vcat(prim_other, prim_rho)
end

"""
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerMulticomponentEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t,
                                               equations::CompressibleEulerMulticomponentEquations2D)
    # Same settings as in `initial_condition`
    RealT = eltype(u)
    c = 2
    A = convert(RealT, 0.1)
    L = 2
    f = 1.0f0 / L
    omega = 2 * convert(RealT, pi) * f

    gamma = totalgamma(u, equations)

    x1, x2 = x
    si, co = sincos((x1 + x2 - t) * omega)
    tmp1 = co * A * omega
    tmp2 = si * A
    tmp3 = gamma - 1
    tmp4 = (2 * c - 1) * tmp3
    tmp5 = (2 * tmp2 * gamma - 2 * tmp2 + tmp4 + 1) * tmp1
    tmp6 = tmp2 + c

    # Here we compute an arbitrary number of different rhos. (one rho is double the next rho while the sum of all rhos is 1
    du_rho = SVector{ncomponents(equations), real(equations)}(2^(i - 1) * (1 - 2) *
                                                              tmp1 / (1 -
                                                               2^ncomponents(equations))
                                                              for i in eachcomponent(equations))

    du1 = tmp5
    du2 = tmp5
    du3 = 2 * ((tmp6 - 1) * tmp3 + tmp6 * gamma) * tmp1

    du_other = SVector(du1, du2, du3)

    return vcat(du_other, du_rho)
end

"""
    initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerMulticomponentEquations2D)

A for multicomponent adapted weak blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t,
                                           equations::CompressibleEulerMulticomponentEquations2D)
    # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
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

    prim_other = SVector(v1, v2, p)

    return prim2cons(vcat(prim_other, prim_rho), equations)
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer,
                      equations::CompressibleEulerMulticomponentEquations2D)
    rho_v1, rho_v2, rho_e = u

    rho = density(u, equations)

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    gamma = totalgamma(u, equations)
    p = (gamma - 1) * (rho_e - 0.5f0 * rho * (v1^2 + v2^2))

    if orientation == 1
        f_rho = densities(u, v1, equations)
        f1 = rho_v1 * v1 + p
        f2 = rho_v2 * v1
        f3 = (rho_e + p) * v1
    else
        f_rho = densities(u, v2, equations)
        f1 = rho_v1 * v2
        f2 = rho_v2 * v2 + p
        f3 = (rho_e + p) * v2
    end

    f_other = SVector(f1, f2, f3)

    return vcat(f_other, f_rho)
end

# Calculate 1D flux for a single point
@inline function flux(u, normal_direction::AbstractVector,
                      equations::CompressibleEulerMulticomponentEquations2D)
    rho_v1, rho_v2, rho_e = u

    rho = density(u, equations)

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
    gamma = totalgamma(u, equations)
    p = (gamma - 1) * (rho_e - 0.5f0 * rho * (v1^2 + v2^2))

    f_rho = densities(u, v_normal, equations)
    f1 = rho_v1 * v_normal + p * normal_direction[1]
    f2 = rho_v2 * v_normal + p * normal_direction[2]
    f3 = (rho_e + p) * v_normal

    f_other = SVector(f1, f2, f3)

    return vcat(f_other, f_rho)
end

"""
    flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations2D)

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
    rhok_avg = SVector{ncomponents(equations), real(equations)}(0.5f0 * (u_ll[i + 3] +
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
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v1_square = 0.5f0 * (v1_ll^2 + v1_rr^2)
    v2_square = 0.5f0 * (v2_ll^2 + v2_rr^2)
    v_sum = v1_avg + v2_avg

    RealT = eltype(u_ll)
    enth = zero(RealT)
    help1_ll = zero(RealT)
    help1_rr = zero(RealT)

    for i in eachcomponent(equations)
        enth += rhok_avg[i] * gas_constants[i]
        help1_ll += u_ll[i + 3] * cv[i]
        help1_rr += u_rr[i + 3] * cv[i]
    end

    T_ll = (rho_e_ll - 0.5f0 * rho_ll * (v1_ll^2 + v2_ll^2)) / help1_ll
    T_rr = (rho_e_rr - 0.5f0 * rho_rr * (v1_rr^2 + v2_rr^2)) / help1_rr
    T = 0.5f0 * (1 / T_ll + 1 / T_rr)
    T_log = ln_mean(1 / T_ll, 1 / T_rr)

    # Calculate fluxes depending on orientation
    help1 = zero(RealT)
    help2 = zero(RealT)
    if orientation == 1
        f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v1_avg
                                                                 for i in eachcomponent(equations))
        for i in eachcomponent(equations)
            help1 += f_rho[i] * cv[i]
            help2 += f_rho[i]
        end
        f1 = (help2) * v1_avg + enth / T
        f2 = (help2) * v2_avg
        f3 = (help1) / T_log - 0.5f0 * (v1_square + v2_square) * (help2) + v1_avg * f1 +
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
        f3 = (help1) / T_log - 0.5f0 * (v1_square + v2_square) * (help2) + v1_avg * f1 +
             v2_avg * f2
    end
    f_other = SVector(f1, f2, f3)

    return vcat(f_other, f_rho)
end

"""
    flux_ranocha(u_ll, u_rr, orientation_or_normal_direction,
                 equations::CompressibleEulerMulticomponentEquations2D)

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
                              equations::CompressibleEulerMulticomponentEquations2D)
    # Unpack left and right state
    @unpack gammas, gas_constants, cv = equations
    rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
    rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
    rhok_mean = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i + 3],
                                                                         u_rr[i + 3])
                                                                 for i in eachcomponent(equations))
    rhok_avg = SVector{ncomponents(equations), real(equations)}(0.5f0 * (u_ll[i + 3] +
                                                                 u_rr[i + 3])
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
    v2_ll = rho_v2_ll / rho_ll
    v2_rr = rho_v2_rr / rho_rr
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    velocity_square_avg = 0.5f0 * (v1_ll * v1_rr + v2_ll * v2_rr)

    # helpful variables
    RealT = eltype(u_ll)
    help1_ll = zero(RealT)
    help1_rr = zero(RealT)
    enth_ll = zero(RealT)
    enth_rr = zero(RealT)
    for i in eachcomponent(equations)
        enth_ll += u_ll[i + 3] * gas_constants[i]
        enth_rr += u_rr[i + 3] * gas_constants[i]
        help1_ll += u_ll[i + 3] * cv[i]
        help1_rr += u_rr[i + 3] * cv[i]
    end

    # temperature and pressure
    T_ll = (rho_e_ll - 0.5f0 * rho_ll * (v1_ll^2 + v2_ll^2)) / help1_ll
    T_rr = (rho_e_rr - 0.5f0 * rho_rr * (v1_rr^2 + v2_rr^2)) / help1_rr
    p_ll = T_ll * enth_ll
    p_rr = T_rr * enth_rr
    p_avg = 0.5f0 * (p_ll + p_rr)
    inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)

    f_rho_sum = zero(RealT)
    if orientation == 1
        f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v1_avg
                                                                 for i in eachcomponent(equations))
        for i in eachcomponent(equations)
            f_rho_sum += f_rho[i]
        end
        f1 = f_rho_sum * v1_avg + p_avg
        f2 = f_rho_sum * v2_avg
        f3 = f_rho_sum * (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one) +
             0.5f0 * (p_ll * v1_rr + p_rr * v1_ll)
    else
        f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v2_avg
                                                                 for i in eachcomponent(equations))
        for i in eachcomponent(equations)
            f_rho_sum += f_rho[i]
        end
        f1 = f_rho_sum * v1_avg
        f2 = f_rho_sum * v2_avg + p_avg
        f3 = f_rho_sum * (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one) +
             0.5f0 * (p_ll * v2_rr + p_rr * v2_ll)
    end

    # momentum and energy flux
    f_other = SVector(f1, f2, f3)

    return vcat(f_other, f_rho)
end

@inline function flux_ranocha(u_ll, u_rr, normal_direction::AbstractVector,
                              equations::CompressibleEulerMulticomponentEquations2D)
    # Unpack left and right state
    @unpack gammas, gas_constants, cv = equations
    rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
    rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
    rhok_mean = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i + 3],
                                                                         u_rr[i + 3])
                                                                 for i in eachcomponent(equations))
    rhok_avg = SVector{ncomponents(equations), real(equations)}(0.5f0 * (u_ll[i + 3] +
                                                                 u_rr[i + 3])
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
    v2_ll = rho_v2_ll / rho_ll
    v2_rr = rho_v2_rr / rho_rr
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    velocity_square_avg = 0.5f0 * (v1_ll * v1_rr + v2_ll * v2_rr)
    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    # helpful variables
    RealT = eltype(u_ll)
    help1_ll = zero(RealT)
    help1_rr = zero(RealT)
    enth_ll = zero(RealT)
    enth_rr = zero(RealT)
    for i in eachcomponent(equations)
        enth_ll += u_ll[i + 3] * gas_constants[i]
        enth_rr += u_rr[i + 3] * gas_constants[i]
        help1_ll += u_ll[i + 3] * cv[i]
        help1_rr += u_rr[i + 3] * cv[i]
    end

    # temperature and pressure
    T_ll = (rho_e_ll - 0.5f0 * rho_ll * (v1_ll^2 + v2_ll^2)) / help1_ll
    T_rr = (rho_e_rr - 0.5f0 * rho_rr * (v1_rr^2 + v2_rr^2)) / help1_rr
    p_ll = T_ll * enth_ll
    p_rr = T_rr * enth_rr
    p_avg = 0.5f0 * (p_ll + p_rr)
    inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)

    f_rho_sum = zero(RealT)
    f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * 0.5f0 *
                                                             (v_dot_n_ll + v_dot_n_rr)
                                                             for i in eachcomponent(equations))
    for i in eachcomponent(equations)
        f_rho_sum += f_rho[i]
    end
    f1 = f_rho_sum * v1_avg + p_avg * normal_direction[1]
    f2 = f_rho_sum * v2_avg + p_avg * normal_direction[2]
    f3 = f_rho_sum * (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one) +
         0.5f0 * (p_ll * v_dot_n_rr + p_rr * v_dot_n_ll)

    # momentum and energy flux
    f_other = SVector(f1, f2, f3)

    return vcat(f_other, f_rho)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::CompressibleEulerMulticomponentEquations2D)
    rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
    rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

    # Get the density and gas gamma
    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)
    gamma_ll = totalgamma(u_ll, equations)
    gamma_rr = totalgamma(u_rr, equations)

    # Get the velocities based on direction
    if orientation == 1
        v_ll = rho_v1_ll / rho_ll
        v_rr = rho_v1_rr / rho_rr
    else # orientation == 2
        v_ll = rho_v2_ll / rho_ll
        v_rr = rho_v2_rr / rho_rr
    end

    # Compute the sound speeds on the left and right
    p_ll = (gamma_ll - 1) * (rho_e_ll - 0.5f0 * (rho_v1_ll^2 + rho_v2_ll^2) / rho_ll)
    c_ll = sqrt(gamma_ll * p_ll / rho_ll)
    p_rr = (gamma_rr - 1) * (rho_e_rr - 0.5f0 * (rho_v1_rr^2 + rho_v2_rr^2) / rho_rr)
    c_rr = sqrt(gamma_rr * p_rr / rho_rr)

    λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end

@inline function max_abs_speeds(u,
                                equations::CompressibleEulerMulticomponentEquations2D)
    rho_v1, rho_v2, rho_e = u

    rho = density(u, equations)
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho

    gamma = totalgamma(u, equations)
    p = (gamma - 1) * (rho_e - 0.5f0 * rho * (v1^2 + v2^2))
    c = sqrt(gamma * p / rho)

    return (abs(v1) + c, abs(v2) + c)
end

@inline function rotate_to_x(u, normal_vector,
                             equations::CompressibleEulerMulticomponentEquations2D)
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D rotation matrix with normal and tangent directions of the form
    # [ n_1  n_2  0    0;
    #   t_1  t_2  0    0;
    #   0    0    1    0
    #   0    0    0    1]
    # where t_1 = -n_2 and t_2 = n_1

    densities = @view u[4:end]
    return SVector(c * u[1] + s * u[2],
                   -s * u[1] + c * u[2],
                   u[3],
                   densities...)
end

# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this back-rotation of the state vector
@inline function rotate_from_x(u, normal_vector,
                               equations::CompressibleEulerMulticomponentEquations2D)
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D back-rotation matrix with normal and tangent directions of the form
    # [ n_1  t_1  0   0;
    #   n_2  t_2  0   0;
    #   0    0    1   0;
    #   0    0    0   1 ]
    # where t_1 = -n_2 and t_2 = n_1

    densities = @view u[4:end]
    return SVector(c * u[1] - s * u[2],
                   s * u[1] + c * u[2],
                   u[3],
                   densities...)
end

# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerMulticomponentEquations2D)
    rho_v1, rho_v2, rho_e = u

    prim_rho = SVector{ncomponents(equations), real(equations)}(u[i + 3]
                                                                for i in eachcomponent(equations))

    rho = density(u, equations)
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    gamma = totalgamma(u, equations)
    p = (gamma - 1) * (rho_e - 0.5f0 * rho * (v1^2 + v2^2))
    prim_other = SVector(v1, v2, p)

    return vcat(prim_other, prim_rho)
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerMulticomponentEquations2D)
    @unpack cv, gammas, gas_constants = equations
    rho_v1, rho_v2, rho_e = u

    rho = density(u, equations)

    # Multicomponent stuff
    RealT = eltype(u)
    help1 = zero(RealT)
    gas_constant = zero(RealT)
    for i in eachcomponent(equations)
        help1 += u[i + 3] * cv[i]
        gas_constant += gas_constants[i] * (u[i + 3] / rho)
    end

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_square = v1^2 + v2^2
    gamma = totalgamma(u, equations)

    p = (gamma - 1) * (rho_e - 0.5f0 * rho * v_square)
    s = log(p) - gamma * log(rho) - log(gas_constant)
    rho_p = rho / p
    T = (rho_e - 0.5f0 * rho * v_square) / (help1)

    entrop_rho = SVector{ncomponents(equations), real(equations)}((cv[i] *
                                                                   (1 - log(T)) +
                                                                   gas_constants[i] *
                                                                   (1 + log(u[i + 3])) -
                                                                   v_square / (2 * T))
                                                                  for i in eachcomponent(equations))

    w1 = gas_constant * v1 * rho_p
    w2 = gas_constant * v2 * rho_p
    w3 = gas_constant * (-rho_p)

    entrop_other = SVector(w1, w2, w3)

    return vcat(entrop_other, entrop_rho)
end

# Convert entropy variables to conservative variables
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
    u2 = rho * v2
    gamma = help1 / help2
    u3 = p / (gamma - 1) + 0.5f0 * rho * v_squared
    cons_other = SVector(u1, u2, u3)
    return vcat(cons_other, cons_rho)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerMulticomponentEquations2D)
    @unpack cv, gammas = equations
    v1, v2, p = prim

    cons_rho = SVector{ncomponents(equations), real(equations)}(prim[i + 3]
                                                                for i in eachcomponent(equations))
    rho = density(prim, equations)
    gamma = totalgamma(prim, equations)

    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_e = p / (gamma - 1) + 0.5f0 * (rho_v1 * v1 + rho_v2 * v2)

    cons_other = SVector(rho_v1, rho_v2, rho_e)

    return vcat(cons_other, cons_rho)
end

@inline function total_entropy(u, equations::CompressibleEulerMulticomponentEquations2D)
    @unpack cv, gammas, gas_constants = equations
    rho = density(u, equations)
    T = temperature(u, equations)

    RealT = eltype(u)
    total_entropy = zero(RealT)
    for i in eachcomponent(equations)
        total_entropy -= u[i + 3] * (cv[i] * log(T) - gas_constants[i] * log(u[i + 3]))
    end

    return total_entropy
end

@inline function temperature(u, equations::CompressibleEulerMulticomponentEquations2D)
    @unpack cv, gammas, gas_constants = equations

    rho_v1, rho_v2, rho_e = u

    rho = density(u, equations)
    RealT = eltype(u)
    help1 = zero(RealT)

    for i in eachcomponent(equations)
        help1 += u[i + 3] * cv[i]
    end

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_square = v1^2 + v2^2
    T = (rho_e - 0.5f0 * rho * v_square) / help1

    return T
end

"""
    totalgamma(u, equations::CompressibleEulerMulticomponentEquations2D)

Function that calculates the total gamma out of all partial gammas using the
partial density fractions as well as the partial specific heats at constant volume.
"""
@inline function totalgamma(u, equations::CompressibleEulerMulticomponentEquations2D)
    @unpack cv, gammas = equations

    RealT = eltype(u)
    help1 = zero(RealT)
    help2 = zero(RealT)

    for i in eachcomponent(equations)
        help1 += u[i + 3] * cv[i] * gammas[i]
        help2 += u[i + 3] * cv[i]
    end

    return help1 / help2
end

@inline function density_pressure(u,
                                  equations::CompressibleEulerMulticomponentEquations2D)
    rho_v1, rho_v2, rho_e = u

    rho = density(u, equations)
    gamma = totalgamma(u, equations)
    rho_times_p = (gamma - 1) * (rho * rho_e - 0.5f0 * (rho_v1^2 + rho_v2^2))

    return rho_times_p
end

@inline function density(u, equations::CompressibleEulerMulticomponentEquations2D)
    RealT = eltype(u)
    rho = zero(RealT)

    for i in eachcomponent(equations)
        rho += u[i + 3]
    end

    return rho
end

@inline function densities(u, v, equations::CompressibleEulerMulticomponentEquations2D)
    return SVector{ncomponents(equations), real(equations)}(u[i + 3] * v
                                                            for i in eachcomponent(equations))
end

@inline function velocity(u, equations::CompressibleEulerMulticomponentEquations2D)
    rho = density(u, equations)
    v1 = u[1] / rho
    v2 = u[2] / rho
    return SVector(v1, v2)
end

@inline function velocity(u, orientation::Int,
                          equations::CompressibleEulerMulticomponentEquations2D)
    rho = density(u, equations)
    v = u[orientation] / rho
    return v
end
end # @muladd
