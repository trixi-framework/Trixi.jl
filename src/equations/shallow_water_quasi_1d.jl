# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    ShallowWaterEquationsQuasi1D(; gravity, H0 = 0, threshold_limiter = nothing threshold_wet = nothing)

The quasi-1D shallow water equations (SWE). The equations are given by
```math
\begin{aligned}
  \frac{\partial}{\partial t}(a h) + \frac{\partial}{\partial x}(a h v) &= 0 \\
    \frac{\partial}{\partial t}(a h v) + \frac{\partial}{\partial x}(a h v^2)
    + g a h \frac{\partial}{\partial x}(h + b) &= 0
\end{aligned}
```
The unknown quantities of the Quasi-1D SWE are the water height ``h`` and the scaled velocity ``v``.
The gravitational constant is denoted by `g`, the (possibly) variable bottom topography function ``b(x)``, and (possibly) variable channel width ``a(x)``. The water height ``h`` is measured from the bottom topography ``b``, therefore one also defines the total water height as ``H = h + b``.

The additional quantity ``H_0`` is also available to store a reference value for the total water height that
is useful to set initial conditions or test the "lake-at-rest" well-balancedness.

Also, there are two thresholds which prevent numerical problems as well as instabilities. Both of them do not
have to be passed, as default values are defined within the struct. The first one, `threshold_limiter`, is
used in [`PositivityPreservingLimiterShallowWater`](@ref) on the water height, as a (small) shift on the initial
condition and cutoff before the next time step. The second one, `threshold_wet`, is applied on the water height to
define when the flow is "wet" before calculating the numerical flux.

The bottom topography function ``b(x)`` and channel width ``a(x)`` are set inside the initial condition routine
for a particular problem setup. To test the conservative form of the SWE one can set the bottom topography
variable `b` to zero and ``a`` to one. 

In addition to the unknowns, Trixi.jl currently stores the bottom topography and channel width values at the approximation points 
despite being fixed in time. This is done for convenience of computing the bottom topography gradients
on the fly during the approximation as well as computing auxiliary quantities like the total water height ``H``
or the entropy variables.
This affects the implementation and use of these equations in various ways:
* The flux values corresponding to the bottom topography and channel width must be zero.
* The bottom topography and channel width values must be included when defining initial conditions, boundary conditions or
  source terms.
* [`AnalysisCallback`](@ref) analyzes this variable.
* Trixi.jl's visualization tools will visualize the bottom topography and channel width by default.
"""
struct ShallowWaterEquationsQuasi1D{RealT <: Real} <:
       AbstractShallowWaterEquations{1, 4}
    gravity::RealT # gravitational constant
    H0::RealT      # constant "lake-at-rest" total water height
    # `threshold_limiter` used in `PositivityPreservingLimiterShallowWater` on water height,
    # as a (small) shift on the initial condition and cutoff before the next time step.
    # Default is 500*eps() which in double precision is ≈1e-13.
    threshold_limiter::RealT
    # `threshold_wet` applied on water height to define when the flow is "wet"
    # before calculating the numerical flux.
    # Default is 5*eps() which in double precision is ≈1e-15.
    threshold_wet::RealT
end

# Allow for flexibility to set the gravitational constant within an elixir depending on the
# application where `gravity_constant=1.0` or `gravity_constant=9.81` are common values.
# The reference total water height H0 defaults to 0.0 but is used for the "lake-at-rest"
# well-balancedness test cases.
# Strict default values for thresholds that performed well in many numerical experiments
function ShallowWaterEquationsQuasi1D(; gravity_constant, H0 = zero(gravity_constant),
                                      threshold_limiter = nothing,
                                      threshold_wet = nothing)
    T = promote_type(typeof(gravity_constant), typeof(H0))
    if threshold_limiter === nothing
        threshold_limiter = 500 * eps(T)
    end
    if threshold_wet === nothing
        threshold_wet = 5 * eps(T)
    end
    ShallowWaterEquationsQuasi1D(gravity_constant, H0, threshold_limiter, threshold_wet)
end

have_nonconservative_terms(::ShallowWaterEquationsQuasi1D) = True()
function varnames(::typeof(cons2cons), ::ShallowWaterEquationsQuasi1D)
    ("a_h", "a_h_v", "b", "a")
end
# Note, we use the total water height, H = h + b, as the first primitive variable for easier
# visualization and setting initial conditions
varnames(::typeof(cons2prim), ::ShallowWaterEquationsQuasi1D) = ("H", "v", "b", "a")

# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_convergence_test(x, t, equations::ShallowWaterEquationsQuasi1D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t,
                                            equations::ShallowWaterEquationsQuasi1D)
    # generates a manufactured solution. 
    # some constants are chosen such that the function is periodic on the domain [0,sqrt(2)]
    Omega = sqrt(2) * pi
    H = 2.0 + 0.5 * sin(Omega * x[1] - t)
    v = 0.25
    b = 0.2 - 0.05 * sin(Omega * x[1])
    a = 1 + 0.1 * cos(Omega * x[1])
    return prim2cons(SVector(H, v, b, a), equations)
end

"""
    source_terms_convergence_test(u, x, t, equations::ShallowWaterEquationsQuasi1D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).

This manufactured solution source term is specifically designed for the bottom topography function
`b(x) = 0.2 - 0.05 * sin(sqrt(2) * pi *x[1])` and channel width 'a(x)= 1 + 0.1 * cos(sqrt(2) * pi * x[1])'
as defined in [`initial_condition_convergence_test`](@ref).
"""
@inline function source_terms_convergence_test(u, x, t,
                                               equations::ShallowWaterEquationsQuasi1D)
    # Same settings as in `initial_condition_convergence_test`. Some derivative simplify because
    # this manufactured solution velocity is taken to be constant
    Omega = sqrt(2) * pi
    H = 2.0 + 0.5 * sin(Omega * x[1] - t)
    H_x = 0.5 * cos(Omega * x[1] - t) * Omega
    H_t = -0.5 * cos(Omega * x[1] - t)

    v = 0.25

    b = 0.2 - 0.05 * sin(Omega * x[1])
    b_x = -0.05 * cos(Omega * x[1]) * Omega

    a = 1 + 0.1 * cos(Omega * x[1])
    a_x = -0.1 * sin(Omega * x[1]) * Omega

    du1 = a * H_t + v * (a_x * (H - b) + a * (H_x - b_x))
    du2 = v * du1 + a * (equations.gravity * (H - b) * H_x)

    return SVector(du1, du2, 0.0, 0.0)
end

# Calculate 1D flux for a single point
# Note, the bottom topography and channel width have no flux
@inline function flux(u, orientation::Integer, equations::ShallowWaterEquationsQuasi1D)
    a_h, a_h_v, _, a = u
    h = waterheight(u, equations)
    v = velocity(u, equations)

    p = 0.5 * a * equations.gravity * h^2

    f1 = a_h_v
    f2 = a_h_v * v + p

    return SVector(f1, f2, zero(eltype(u)), zero(eltype(u)))
end

"""
    flux_nonconservative_chan_etal(u_ll, u_rr, orientation::Integer,
                                   equations::ShallowWaterEquationsQuasi1D)

Non-symmetric two-point volume flux discretizing the nonconservative (source) term
that contains the gradient of the bottom topography [`ShallowWaterEquationsQuasi1D`](@ref) 
and the channel width.

Further details are available in the paper:
- Jesse Chan, Khemraj Shukla, Xinhui Wu, Ruofeng Liu, Prani Nalluri (2023)
    High order entropy stable schemes for the quasi-one-dimensional
    shallow water and compressible Euler equations
    [DOI: 10.48550/arXiv.2307.12089](https://doi.org/10.48550/arXiv.2307.12089)
"""
@inline function flux_nonconservative_chan_etal(u_ll, u_rr, orientation::Integer,
                                                equations::ShallowWaterEquationsQuasi1D)
    a_h_ll, _, b_ll, a_ll = u_ll
    a_h_rr, _, b_rr, a_rr = u_rr

    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)

    z = zero(eltype(u_ll))

    return SVector(z, equations.gravity * a_ll * h_ll * (h_rr + b_rr), z, z)
end

"""
    flux_chan_etal(u_ll, u_rr, orientation,
                   equations::ShallowWaterEquationsQuasi1D)

Total energy conservative (mathematical entropy for quasi 1D shallow water equations) split form.
When the bottom topography is nonzero this scheme will be well-balanced when used as a `volume_flux`.
The `surface_flux` should still use, e.g., [`FluxPlusDissipation(flux_chan_etal, DissipationLocalLaxFriedrichs())`](@ref).

Further details are available in the paper:
- Jesse Chan, Khemraj Shukla, Xinhui Wu, Ruofeng Liu, Prani Nalluri (2023) 
  High order entropy stable schemes for the quasi-one-dimensional
  shallow water and compressible Euler equations
  [DOI: 10.48550/arXiv.2307.12089](https://doi.org/10.48550/arXiv.2307.12089)
"""
@inline function flux_chan_etal(u_ll, u_rr, orientation::Integer,
                                equations::ShallowWaterEquationsQuasi1D)
    a_h_ll, a_h_v_ll, _, _ = u_ll
    a_h_rr, a_h_v_rr, _, _ = u_rr

    v_ll = velocity(u_ll, equations)
    v_rr = velocity(u_rr, equations)

    f1 = 0.5 * (a_h_v_ll + a_h_v_rr)
    f2 = f1 * 0.5 * (v_ll + v_rr)

    return SVector(f1, f2, zero(eltype(u_ll)), zero(eltype(u_ll)))
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::ShallowWaterEquationsQuasi1D)
    # Get the velocity quantities
    v_ll = velocity(u_ll, equations)
    v_rr = velocity(u_rr, equations)

    # Calculate the wave celerity on the left and right
    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)
    c_ll = sqrt(equations.gravity * h_ll)
    c_rr = sqrt(equations.gravity * h_rr)

    return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end

# Specialized `DissipationLocalLaxFriedrichs` to avoid spurious dissipation in the bottom topography
# and channel width
@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr,
                                                              orientation_or_normal_direction,
                                                              equations::ShallowWaterEquationsQuasi1D)
    λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction,
                                  equations)
    diss = -0.5 * λ * (u_rr - u_ll)
    return SVector(diss[1], diss[2], zero(eltype(u_ll)), zero(eltype(u_ll)))
end

@inline function max_abs_speeds(u, equations::ShallowWaterEquationsQuasi1D)
    h = waterheight(u, equations)
    v = velocity(u, equations)

    c = equations.gravity * sqrt(h)
    return (abs(v) + c,)
end

# Helper function to extract the velocity vector from the conservative variables
@inline function velocity(u, equations::ShallowWaterEquationsQuasi1D)
    a_h, a_h_v, _, _ = u

    v = a_h_v / a_h

    return v
end

# Convert conservative variables to primitive
@inline function cons2prim(u, equations::ShallowWaterEquationsQuasi1D)
    a_h, _, b, a = u
    h = a_h / a
    H = h + b
    v = velocity(u, equations)
    return SVector(H, v, b, a)
end

# Convert conservative variables to entropy variables
# Note, only the first two are the entropy variables, the third and fourth entries still
# just carry the bottom topography and channel width values for convenience
@inline function cons2entropy(u, equations::ShallowWaterEquationsQuasi1D)
    a_h, a_h_v, b, a = u
    h = waterheight(u, equations)
    v = velocity(u, equations)
    #entropy variables are the same as ones in standard shallow water equations
    w1 = equations.gravity * (h + b) - 0.5 * v^2
    w2 = v

    return SVector(w1, w2, b, a)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::ShallowWaterEquationsQuasi1D)
    H, v, b, a = prim

    a_h = a * (H - b)
    a_h_v = a_h * v
    return SVector(a_h, a_h_v, b, a)
end

@inline function waterheight(u, equations::ShallowWaterEquationsQuasi1D)
    return u[1] / u[4]
end

# Entropy function for the shallow water equations is the total energy
@inline function entropy(cons, equations::ShallowWaterEquationsQuasi1D)
    a = cons[4]
    return a * energy_total(cons, equations)
end

# Calculate total energy for a conservative state `cons`
@inline function energy_total(cons, equations::ShallowWaterEquationsQuasi1D)
    a_h, a_h_v, b, a = cons
    e = (a_h_v^2) / (2 * a * a_h) + 0.5 * equations.gravity * (a_h^2 / a) +
        equations.gravity * a_h * b
    return e
end

# Calculate the error for the "lake-at-rest" test case where H = h+b should
# be a constant value over time. Note, assumes there is a single reference
# water height `H0` with which to compare.
#
# TODO: TrixiShallowWater: where should `threshold_limiter` live? May need
# to modify or have different versions of the `lake_at_rest_error` function
@inline function lake_at_rest_error(u, equations::ShallowWaterEquationsQuasi1D)
    _, _, b, _ = u
    h = waterheight(u, equations)

    # For well-balancedness testing with possible wet/dry regions the reference
    # water height `H0` accounts for the possibility that the bottom topography
    # can emerge out of the water as well as for the threshold offset to avoid
    # division by a "hard" zero water heights as well.
    H0_wet_dry = max(equations.H0, b + equations.threshold_limiter)

    return abs(H0_wet_dry - (h + b))
end
end # @muladd
