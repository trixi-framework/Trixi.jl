# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    CompressibleEulerEquationsQuasi1D(gamma)

The quasi-1d compressible Euler equations (see Chan et al.  [DOI: 10.48550/arXiv.2307.12089](https://doi.org/10.48550/arXiv.2307.12089)  for details)
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
a \rho \\ a \rho v_1 \\ a e
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
a \rho v_1 \\ a \rho v_1^2 \\ a v_1 (e +p)
\end{pmatrix}
+ 
a \frac{\partial}{\partial x}
\begin{pmatrix}
0 \\ p \\ 0    
\end{pmatrix}
=
\begin{pmatrix}
0 \\ 0 \\ 0
\end{pmatrix}
```
for an ideal gas with ratio of specific heats `gamma` in one space dimension.
Here, ``\rho`` is the density, ``v_1`` the velocity, ``e`` the specific total energy **rather than** specific internal energy, 
``a`` the (possibly) variable nozzle width, and
```math
p = (\gamma - 1) \left( e - \frac{1}{2} \rho v_1^2 \right)
```
the pressure.

The nozzle width function ``a(x)`` is set inside the initial condition routine
for a particular problem setup. To test the conservative form of the compressible Euler equations one can set the 
nozzle width variable ``a`` to one. 

In addition to the unknowns, Trixi.jl currently stores the nozzle width values at the approximation points 
despite being fixed in time.
This affects the implementation and use of these equations in various ways:
* The flux values corresponding to the nozzle width must be zero.
* The nozzle width values must be included when defining initial conditions, boundary conditions or
  source terms.
* [`AnalysisCallback`](@ref) analyzes this variable.
* Trixi.jl's visualization tools will visualize the nozzle width by default.
"""
struct CompressibleEulerEquationsQuasi1D{RealT <: Real} <:
       AbstractCompressibleEulerEquations{1, 4}
    gamma::RealT               # ratio of specific heats
    inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications

    function CompressibleEulerEquationsQuasi1D(gamma)
        γ, inv_gamma_minus_one = promote(gamma, inv(gamma - 1))
        new{typeof(γ)}(γ, inv_gamma_minus_one)
    end
end

have_nonconservative_terms(::CompressibleEulerEquationsQuasi1D) = True()
function varnames(::typeof(cons2cons), ::CompressibleEulerEquationsQuasi1D)
    ("a_rho", "a_rho_v1", "a_e", "a")
end
function varnames(::typeof(cons2prim), ::CompressibleEulerEquationsQuasi1D)
    ("rho", "v1", "p", "a")
end

"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerEquationsQuasi1D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t,
                                            equations::CompressibleEulerEquationsQuasi1D)
    RealT = eltype(x)
    c = 2
    A = convert(RealT, 0.1)
    L = 2
    f = 1.0f0 / L
    ω = 2 * convert(RealT, pi) * f
    ini = c + A * sin(ω * (x[1] - t))

    rho = ini
    v1 = 1
    e = ini^2 / rho
    p = (equations.gamma - 1) * (e - 0.5f0 * rho * v1^2)
    a = 1.5f0 - 0.5f0 * cos(x[1] * convert(RealT, pi))

    return prim2cons(SVector(rho, v1, p, a), equations)
end

"""
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerEquationsQuasi1D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).

This manufactured solution source term is specifically designed for the mozzle width 'a(x) = 1.5 - 0.5 * cos(x[1] * pi)'
as defined in [`initial_condition_convergence_test`](@ref).
"""
@inline function source_terms_convergence_test(u, x, t,
                                               equations::CompressibleEulerEquationsQuasi1D)
    # Same settings as in `initial_condition_convergence_test`. 
    # Derivatives calculated with ForwardDiff.jl
    RealT = eltype(u)
    c = 2
    A = convert(RealT, 0.1)
    L = 2
    f = 1.0f0 / L
    ω = 2 * convert(RealT, pi) * f
    x1, = x
    ini(x1, t) = c + A * sin(ω * (x1 - t))

    rho(x1, t) = ini(x1, t)
    v1(x1, t) = 1
    e(x1, t) = ini(x1, t)^2 / rho(x1, t)
    p1(x1, t) = (equations.gamma - 1) * (e(x1, t) - 0.5f0 * rho(x1, t) * v1(x1, t)^2)
    a(x1, t) = 1.5f0 - 0.5f0 * cos(x1 * pi)

    arho(x1, t) = a(x1, t) * rho(x1, t)
    arhou(x1, t) = arho(x1, t) * v1(x1, t)
    aE(x1, t) = a(x1, t) * e(x1, t)

    darho_dt(x1, t) = ForwardDiff.derivative(t -> arho(x1, t), t)
    darhou_dx(x1, t) = ForwardDiff.derivative(x1 -> arhou(x1, t), x1)

    arhouu(x1, t) = arhou(x1, t) * v1(x1, t)
    darhou_dt(x1, t) = ForwardDiff.derivative(t -> arhou(x1, t), t)
    darhouu_dx(x1, t) = ForwardDiff.derivative(x1 -> arhouu(x1, t), x1)
    dp1_dx(x1, t) = ForwardDiff.derivative(x1 -> p1(x1, t), x1)

    auEp(x1, t) = a(x1, t) * v1(x1, t) * (e(x1, t) + p1(x1, t))
    daE_dt(x1, t) = ForwardDiff.derivative(t -> aE(x1, t), t)
    dauEp_dx(x1, t) = ForwardDiff.derivative(x1 -> auEp(x1, t), x1)

    du1 = darho_dt(x1, t) + darhou_dx(x1, t)
    du2 = darhou_dt(x1, t) + darhouu_dx(x1, t) + a(x1, t) * dp1_dx(x1, t)
    du3 = daE_dt(x1, t) + dauEp_dx(x1, t)

    return SVector(du1, du2, du3, 0)
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer,
                      equations::CompressibleEulerEquationsQuasi1D)
    a_rho, a_rho_v1, a_e, a = u
    rho, v1, p, a = cons2prim(u, equations)
    e = a_e / a

    # Ignore orientation since it is always "1" in 1D
    f1 = a_rho_v1
    f2 = a_rho_v1 * v1
    f3 = a * v1 * (e + p)

    return SVector(f1, f2, f3, 0)
end

"""
    flux_nonconservative_chan_etal(u_ll, u_rr, orientation::Integer,
                                   equations::CompressibleEulerEquationsQuasi1D)
    flux_nonconservative_chan_etal(u_ll, u_rr, normal_direction, 
                                   equations::CompressibleEulerEquationsQuasi1D)
    flux_nonconservative_chan_etal(u_ll, u_rr, normal_ll, normal_rr,
                                   equations::CompressibleEulerEquationsQuasi1D)

Non-symmetric two-point volume flux discretizing the nonconservative (source) term
that contains the gradient of the pressure  [`CompressibleEulerEquationsQuasi1D`](@ref) 
and the nozzle width.

Further details are available in the paper:
- Jesse Chan, Khemraj Shukla, Xinhui Wu, Ruofeng Liu, Prani Nalluri (2023)
    High order entropy stable schemes for the quasi-one-dimensional
    shallow water and compressible Euler equations
    [DOI: 10.48550/arXiv.2307.12089](https://doi.org/10.48550/arXiv.2307.12089)    
"""
@inline function flux_nonconservative_chan_etal(u_ll, u_rr, orientation::Integer,
                                                equations::CompressibleEulerEquationsQuasi1D)
    #Variables
    _, _, p_ll, a_ll = cons2prim(u_ll, equations)
    _, _, p_rr, _ = cons2prim(u_rr, equations)

    # For flux differencing using non-conservative terms, we return the 
    # non-conservative flux scaled by 2. This cancels with a factor of 0.5 
    # in the arithmetic average of {p}.
    p_avg = p_ll + p_rr

    return SVector(0, a_ll * p_avg, 0, 0)
end

# While `normal_direction` isn't strictly necessary in 1D, certain solvers assume that 
# the normal component is incorporated into the numerical flux. 
# 
# See `flux(u, normal_direction::AbstractVector, equations::AbstractEquations{1})` for a 
# similar implementation.
@inline function flux_nonconservative_chan_etal(u_ll, u_rr,
                                                normal_direction::AbstractVector,
                                                equations::CompressibleEulerEquationsQuasi1D)
    return normal_direction[1] *
           flux_nonconservative_chan_etal(u_ll, u_rr, 1, equations)
end

@inline function flux_nonconservative_chan_etal(u_ll, u_rr,
                                                normal_ll::AbstractVector,
                                                normal_rr::AbstractVector,
                                                equations::CompressibleEulerEquationsQuasi1D)
    # normal_ll should be equal to normal_rr in 1D
    return flux_nonconservative_chan_etal(u_ll, u_rr, normal_ll, equations)
end

"""
@inline function flux_chan_etal(u_ll, u_rr, orientation::Integer,
                                           equations::CompressibleEulerEquationsQuasi1D)

Conservative (symmetric) part of the entropy conservative flux for quasi 1D compressible Euler equations split form.
This flux is a generalization of [`flux_ranocha`](@ref) for [`CompressibleEulerEquations1D`](@ref).
Further details are available in the paper:
- Jesse Chan, Khemraj Shukla, Xinhui Wu, Ruofeng Liu, Prani Nalluri (2023) 
  High order entropy stable schemes for the quasi-one-dimensional
  shallow water and compressible Euler equations
  [DOI: 10.48550/arXiv.2307.12089](https://doi.org/10.48550/arXiv.2307.12089)     
"""
@inline function flux_chan_etal(u_ll, u_rr, orientation::Integer,
                                equations::CompressibleEulerEquationsQuasi1D)
    # Unpack left and right state
    rho_ll, v1_ll, p_ll, a_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, p_rr, a_rr = cons2prim(u_rr, equations)

    # Compute the necessary mean values
    rho_mean = ln_mean(rho_ll, rho_rr)
    # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
    # in exact arithmetic since
    #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
    #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
    inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    a_v1_avg = 0.5f0 * (a_ll * v1_ll + a_rr * v1_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    velocity_square_avg = 0.5f0 * (v1_ll * v1_rr)

    # Calculate fluxes
    # Ignore orientation since it is always "1" in 1D
    f1 = rho_mean * a_v1_avg
    f2 = rho_mean * a_v1_avg * v1_avg
    f3 = f1 * (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one) +
         0.5f0 * (p_ll * a_rr * v1_rr + p_rr * a_ll * v1_ll)

    return SVector(f1, f2, f3, 0)
end

# While `normal_direction` isn't strictly necessary in 1D, certain solvers assume that 
# the normal component is incorporated into the numerical flux. 
# 
# See `flux(u, normal_direction::AbstractVector, equations::AbstractEquations{1})` for a 
# similar implementation.
@inline function flux_chan_etal(u_ll, u_rr, normal_direction::AbstractVector,
                                equations::CompressibleEulerEquationsQuasi1D)
    return normal_direction[1] * flux_chan_etal(u_ll, u_rr, 1, equations)
end

# Calculate estimates for maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::CompressibleEulerEquationsQuasi1D)
    a_rho_ll, a_rho_v1_ll, a_e_ll, a_ll = u_ll
    a_rho_rr, a_rho_v1_rr, a_e_rr, a_rr = u_rr

    # Calculate primitive variables and speed of sound
    rho_ll = a_rho_ll / a_ll
    e_ll = a_e_ll / a_ll
    v1_ll = a_rho_v1_ll / a_rho_ll
    v_mag_ll = abs(v1_ll)
    p_ll = (equations.gamma - 1) * (e_ll - 0.5f0 * rho_ll * v_mag_ll^2)
    c_ll = sqrt(equations.gamma * p_ll / rho_ll)
    rho_rr = a_rho_rr / a_rr
    e_rr = a_e_rr / a_rr
    v1_rr = a_rho_v1_rr / a_rho_rr
    v_mag_rr = abs(v1_rr)
    p_rr = (equations.gamma - 1) * (e_rr - 0.5f0 * rho_rr * v_mag_rr^2)
    c_rr = sqrt(equations.gamma * p_rr / rho_rr)

    λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end

@inline function max_abs_speeds(u, equations::CompressibleEulerEquationsQuasi1D)
    a_rho, a_rho_v1, a_e, a = u
    rho = a_rho / a
    v1 = a_rho_v1 / a_rho
    e = a_e / a
    p = (equations.gamma - 1) * (e - 0.5f0 * rho * v1^2)
    c = sqrt(equations.gamma * p / rho)

    return (abs(v1) + c,)
end

# Convert conservative variables to primitive. We use the convention that the primitive
# variables for the quasi-1D equations are `(rho, v1, p)` (i.e., the same as the primitive
# variables for `CompressibleEulerEquations1D`)
@inline function cons2prim(u, equations::CompressibleEulerEquationsQuasi1D)
    a_rho, a_rho_v1, a_e, a = u
    q = cons2prim(SVector(a_rho, a_rho_v1, a_e) / a,
                  CompressibleEulerEquations1D(equations.gamma))

    return SVector(q[1], q[2], q[3], a)
end

# The entropy for the quasi-1D compressible Euler equations is the entropy for the
# 1D compressible Euler equations scaled by the channel width `a`.
@inline function entropy(u, equations::CompressibleEulerEquationsQuasi1D)
    a_rho, a_rho_v1, a_e, a = u
    return a * entropy(SVector(a_rho, a_rho_v1, a_e) / a,
                   CompressibleEulerEquations1D(equations.gamma))
end

# Convert conservative variables to entropy. The entropy variables for the 
# quasi-1D compressible Euler equations are identical to the entropy variables
# for the standard Euler equations for an appropriate definition of `entropy`.
@inline function cons2entropy(u, equations::CompressibleEulerEquationsQuasi1D)
    a_rho, a_rho_v1, a_e, a = u
    w = cons2entropy(SVector(a_rho, a_rho_v1, a_e) / a,
                     CompressibleEulerEquations1D(equations.gamma))

    # we follow the convention for other spatially-varying equations such as
    # `ShallowWaterEquations1D` and return the spatially varying coefficient 
    # `a` as the final entropy variable.
    return SVector(w[1], w[2], w[3], a)
end

# Convert primitive to conservative variables
@inline function prim2cons(u, equations::CompressibleEulerEquationsQuasi1D)
    rho, v1, p, a = u
    q = prim2cons(u, CompressibleEulerEquations1D(equations.gamma))

    return SVector(a * q[1], a * q[2], a * q[3], a)
end

@inline function density(u, equations::CompressibleEulerEquationsQuasi1D)
    a_rho, _, _, a = u
    rho = a_rho / a
    return rho
end

@inline function pressure(u, equations::CompressibleEulerEquationsQuasi1D)
    a_rho, a_rho_v1, a_e, a = u
    return pressure(SVector(a_rho, a_rho_v1, a_e) / a,
                    CompressibleEulerEquations1D(equations.gamma))
end

@inline function density_pressure(u, equations::CompressibleEulerEquationsQuasi1D)
    a_rho, a_rho_v1, a_e, a = u
    return density_pressure(SVector(a_rho, a_rho_v1, a_e) / a,
                            CompressibleEulerEquations1D(equations.gamma))
end
end # @muladd
