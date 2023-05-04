# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    ShallowWaterEquations1D(gravity, H0)

Shallow water equations (SWE) in one space dimension. The equations are given by
```math
\begin{aligned}
  \frac{\partial h}{\partial t} + \frac{\partial}{\partial x}(h v) &= 0 \\
    \frac{\partial}{\partial t}(h v) + \frac{\partial}{\partial x}\left(h v^2 + \frac{g}{2}h^2\right)
    + g h \frac{\partial b}{\partial x} &= 0
\end{aligned}
```
The unknown quantities of the SWE are the water height ``h`` and the velocity ``v``.
The gravitational constant is denoted by `g` and the (possibly) variable bottom topography function ``b(x)``.
Conservative variable water height ``h`` is measured from the bottom topography ``b``, therefore one
also defines the total water height as ``H = h + b``.

The additional quantity ``H_0`` is also available to store a reference value for the total water height that
is useful to set initial conditions or test the "lake-at-rest" well-balancedness.

The bottom topography function ``b(x)`` is set inside the initial condition routine
for a particular problem setup. To test the conservative form of the SWE one can set the bottom topography
variable `b` to zero.

In addition to the unknowns, Trixi.jl currently stores the bottom topography values at the approximation points
despite being fixed in time. This is done for convenience of computing the bottom topography gradients
on the fly during the approximation as well as computing auxiliary quantities like the total water height ``H``
or the entropy variables.
This affects the implementation and use of these equations in various ways:
* The flux values corresponding to the bottom topography must be zero.
* The bottom topography values must be included when defining initial conditions, boundary conditions or
  source terms.
* [`AnalysisCallback`](@ref) analyzes this variable.
* Trixi.jl's visualization tools will visualize the bottom topography by default.

References for the SWE are many but a good introduction is available in Chapter 13 of the book:
- Randall J. LeVeque (2002)
  Finite Volume Methods for Hyperbolic Problems
  [DOI: 10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253)
"""
struct ShallowWaterEquations1D{RealT<:Real} <: AbstractShallowWaterEquations{1, 3}
  gravity::RealT # gravitational constant
  H0::RealT      # constant "lake-at-rest" total water height
end

# Allow for flexibility to set the gravitational constant within an elixir depending on the
# application where `gravity_constant=1.0` or `gravity_constant=9.81` are common values.
# The reference total water height H0 defaults to 0.0 but is used for the "lake-at-rest"
# well-balancedness test cases
function ShallowWaterEquations1D(; gravity_constant, H0=0.0)
  ShallowWaterEquations1D(gravity_constant, H0)
end


have_nonconservative_terms(::ShallowWaterEquations1D) = True()
varnames(::typeof(cons2cons), ::ShallowWaterEquations1D) = ("h", "h_v", "b")
# Note, we use the total water height, H = h + b, as the first primitive variable for easier
# visualization and setting initial conditions
varnames(::typeof(cons2prim), ::ShallowWaterEquations1D) = ("H", "v", "b")


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_convergence_test(x, t, equations::ShallowWaterEquations1D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""

function initial_condition_convergence_test(x, t, equations::ShallowWaterEquations1D)
  # some constants are chosen such that the function is periodic on the domain [0,sqrt(2)]
  c  = 7.0
  omega_x = 2.0 * pi * sqrt(2.0)
  omega_t = 2.0 * pi

  H = c + cos(omega_x * x[1]) * cos(omega_t * t)
  v = 0.5
  b = 2.0 + 0.5 * sin(sqrt(2.0) * pi * x[1])
  return prim2cons(SVector(H, v, b), equations)
end

"""
    source_terms_convergence_test(u, x, t, equations::ShallowWaterEquations1D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).

This manufactured solution source term is specifically designed for the bottom topography function
`b(x) = 2.0 + 0.5 * sin(sqrt(2.0)*pi*x[1])`
as defined in [`initial_condition_convergence_test`](@ref).
"""

@inline function source_terms_convergence_test(u, x, t, equations::ShallowWaterEquations1D)
  # Same settings as in `initial_condition_convergence_test`. Some derivative simplify because
  # this manufactured solution velocity is taken to be constant
  c  = 7.0
  omega_x = 2.0 * pi * sqrt(2.0)
  omega_t = 2.0 * pi
  omega_b = sqrt(2.0) * pi
  v = 0.5

  sinX, cosX = sincos(omega_x * x[1])
  sinT, cosT = sincos(omega_t * t )

  H = c + cosX * cosT
  H_x = -omega_x * sinX * cosT
  # this time derivative for the water height exploits that the bottom topography is
  # fixed in time such that H_t = (h+b)_t = h_t + 0
  H_t = -omega_t * cosX * sinT

  # bottom topography and its spatial derivative
  b = 2.0 + 0.5 * sin(sqrt(2.0) * pi * x[1])
  b_x = 0.5 * omega_b * cos(omega_b * x[1])

  du1 = H_t + v * (H_x - b_x)
  du2 = v * du1 + equations.gravity * (H - b) * H_x
  return SVector(du1, du2, 0.0)
end

"""
    initial_condition_weak_blast_wave(x, t, equations::ShallowWaterEquations1D)

A weak blast wave discontinuity useful for testing, e.g., total energy conservation.
Note for the shallow water equations to the total energy acts as a mathematical entropy function.
"""
function initial_condition_weak_blast_wave(x, t, equations::ShallowWaterEquations1D)

  inicenter = 0.7
  x_norm = x[1] - inicenter
  r = abs(x_norm)

  # Calculate primitive variables
  H = r > 0.5 ? 3.25 : 4.0
  v = r > 0.5 ? 0.0 : 0.1882
  b = sin(x[1]) # arbitrary continuous function

  return prim2cons(SVector(H, v, b), equations)
end

"""
    boundary_condition_slip_wall(u_inner, orientation_or_normal, x, t, surface_flux_function,
                                  equations::ShallowWaterEquations1D)

Create a boundary state by reflecting the normal velocity component and keep
the tangential velocity component unchanged. The boundary water height is taken from
the internal value.

For details see Section 9.2.5 of the book:
- Eleuterio F. Toro (2001)
  Shock-Capturing Methods for Free-Surface Shallow Flows
  1st edition
  ISBN 0471987662
"""
@inline function boundary_condition_slip_wall(u_inner, orientation_or_normal, direction,
                                              x, t,
                                              surface_flux_function,
                                              equations::ShallowWaterEquations1D)

  # create the "external" boundary solution state
  u_boundary = SVector(u_inner[1],
                       -u_inner[2],
                       u_inner[3])

  # calculate the boundary flux
  if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation_or_normal, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation_or_normal, equations)
  end

  return flux
end

# Calculate 1D flux for a single point
# Note, the bottom topography has no flux
@inline function flux(u, orientation::Integer, equations::ShallowWaterEquations1D)
  h, h_v, _ = u
  v = velocity(u, equations)

  p = 0.5 * equations.gravity * h^2

  f1 = h_v
  f2 = h_v * v + p

  return SVector(f1, f2, zero(eltype(u)))
end

"""
    flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                          equations::ShallowWaterEquations1D)

Non-symmetric two-point volume flux discretizing the nonconservative (source) term
that contains the gradient of the bottom topography [`ShallowWaterEquations1D`](@ref).

Further details are available in the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                                       equations::ShallowWaterEquations1D)
  # Pull the necessary left and right state information
  h_ll = waterheight(u_ll, equations)
  b_rr = u_rr[3]

  z = zero(eltype(u_ll))

  # Bottom gradient nonconservative term: (0, g h b_x, 0)
  f = SVector(z, equations.gravity * h_ll * b_rr, z)

  return f
end

"""
    flux_nonconservative_fjordholm_etal(u_ll, u_rr, orientation::Integer,
                                        equations::ShallowWaterEquations1D)

Non-symmetric two-point surface flux discretizing the nonconservative (source) term of
that contains the gradient of the bottom topography [`ShallowWaterEquations1D`](@ref).

This contains additional terms compared to [`flux_nonconservative_wintermeyer_etal`](@ref)
that account for possible discontinuities in the bottom topography function.
Thus, this flux should be used in general at interfaces. For flux differencing volume terms,
[`flux_nonconservative_wintermeyer_etal`](@ref) is analytically equivalent but slightly
cheaper.

Further details for the original finite volume formulation are available in
- Ulrik S. Fjordholm, Siddhartha Mishr and Eitan Tadmor (2011)
  Well-balanced and energy stable schemes for the shallow water equations with discontinuous topography
  [DOI: 10.1016/j.jcp.2011.03.042](https://doi.org/10.1016/j.jcp.2011.03.042)
and for curvilinear 2D case in the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_nonconservative_fjordholm_etal(u_ll, u_rr, orientation::Integer,
                                                     equations::ShallowWaterEquations1D)
  # Pull the necessary left and right state information
  h_ll, _, b_ll = u_ll
  h_rr, _, b_rr = u_rr

  h_average = 0.5 * (h_ll + h_rr)
  b_jump = b_rr - b_ll

  # Includes two parts:
  #  (i)  Diagonal (consistent) term from the volume flux that uses `b_ll` to avoid
  #       cross-averaging across a discontinuous bottom topography
  #  (ii) True surface part that uses `h_average` and `b_jump` to handle discontinuous bathymetry
  z = zero(eltype(u_ll))

  f = SVector(z,
              equations.gravity * h_ll * b_ll + equations.gravity * h_average * b_jump,
              z)

  return f
end


"""
    flux_nonconservative_audusse_etal(u_ll, u_rr, orientation::Integer,
                                      equations::ShallowWaterEquations1D)

Non-symmetric two-point surface flux that discretizes the nonconservative (source) term.
The discretization uses the `hydrostatic_reconstruction_audusse_etal` on the conservative
variables.

This hydrostatic reconstruction ensures that the finite volume numerical fluxes remain
well-balanced for discontinuous bottom topographies [`ShallowWaterEquations1D`](@ref).
Should be used together with [`FluxHydrostaticReconstruction`](@ref) and
[`hydrostatic_reconstruction_audusse_etal`](@ref) in the surface flux to ensure consistency.

Further details on the hydrostatic reconstruction and its motivation can be found in
- Emmanuel Audusse, François Bouchut, Marie-Odile Bristeau, Rupert Klein, and Benoit Perthame (2004)
  A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows
  [DOI: 10.1137/S1064827503431090](https://doi.org/10.1137/S1064827503431090)
"""
@inline function flux_nonconservative_audusse_etal(u_ll, u_rr,
                                                   orientation::Integer,
                                                   equations::ShallowWaterEquations1D)
  # Pull the water height and bottom topography on the left
  h_ll, _, b_ll = u_ll

  # Create the hydrostatic reconstruction for the left solution state
  u_ll_star, _ = hydrostatic_reconstruction_audusse_etal(u_ll, u_rr, equations)

  # Copy the reconstructed water height for easier to read code
  h_ll_star = u_ll_star[1]

  z = zero(eltype(u_ll))
  # Includes two parts:
  #   (i)  Diagonal (consistent) term from the volume flux that uses `b_ll` to avoid
  #        cross-averaging across a discontinuous bottom topography
  #   (ii) True surface part that uses `h_ll` and `h_ll_star` to handle discontinuous bathymetry
  return SVector(z,
                 equations.gravity * h_ll * b_ll + equations.gravity * ( h_ll^2 - h_ll_star^2 ),
                 z)
end


"""
    flux_fjordholm_etal(u_ll, u_rr, orientation,
                        equations::ShallowWaterEquations1D)

Total energy conservative (mathematical entropy for shallow water equations). When the bottom topography
is nonzero this should only be used as a surface flux otherwise the scheme will not be well-balanced.
For well-balancedness in the volume flux use [`flux_wintermeyer_etal`](@ref).

Details are available in Eq. (4.1) in the paper:
- Ulrik S. Fjordholm, Siddhartha Mishr and Eitan Tadmor (2011)
  Well-balanced and energy stable schemes for the shallow water equations with discontinuous topography
  [DOI: 10.1016/j.jcp.2011.03.042](https://doi.org/10.1016/j.jcp.2011.03.042)
"""
@inline function flux_fjordholm_etal(u_ll, u_rr, orientation::Integer, equations::ShallowWaterEquations1D)
  # Unpack left and right state
  h_ll = waterheight(u_ll, equations)
  v_ll = velocity(u_ll, equations)
  h_rr = waterheight(u_rr, equations)
  v_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  h_avg = 0.5 * (h_ll   + h_rr  )
  v_avg = 0.5 * (v_ll  + v_rr )
  p_avg = 0.25 * equations.gravity * (h_ll^2 + h_rr^2)

  # Calculate fluxes depending on orientation
  f1 = h_avg * v_avg
  f2 = f1 * v_avg + p_avg

  return SVector(f1, f2, zero(eltype(u_ll)))
end

"""
    flux_wintermeyer_etal(u_ll, u_rr, orientation,
                          equations::ShallowWaterEquations1D)

Total energy conservative (mathematical entropy for shallow water equations) split form.
When the bottom topography is nonzero this scheme will be well-balanced when used as a `volume_flux`.
The `surface_flux` should still use, e.g., [`flux_fjordholm_etal`](@ref).

Further details are available in Theorem 1 of the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_wintermeyer_etal(u_ll, u_rr, orientation::Integer, equations::ShallowWaterEquations1D)
  # Unpack left and right state
  h_ll, h_v_ll, _ = u_ll
  h_rr, h_v_rr, _ = u_rr

  # Get the velocities on either side
  v_ll = velocity(u_ll, equations)
  v_rr = velocity(u_rr, equations)

  # Average each factor of products in flux
  v_avg = 0.5 * (v_ll + v_rr)
  p_avg = 0.5 * equations.gravity * h_ll * h_rr

  # Calculate fluxes depending on orientation
  f1 = 0.5 * (h_v_ll + h_v_rr)
  f2 = f1 * v_avg + p_avg

  return SVector(f1, f2, zero(eltype(u_ll)))
end


"""
    hydrostatic_reconstruction_audusse_etal(u_ll, u_rr, orientation::Integer,
                                            equations::ShallowWaterEquations1D)

A particular type of hydrostatic reconstruction on the water height to guarantee well-balancedness
for a general bottom topography [`ShallowWaterEquations1D`](@ref). The reconstructed solution states
`u_ll_star` and `u_rr_star` variables are used to evaluate the surface numerical flux at the interface.
Use in combination with the generic numerical flux routine [`FluxHydrostaticReconstruction`](@ref).

Further details on this hydrostatic reconstruction and its motivation can be found in
- Emmanuel Audusse, François Bouchut, Marie-Odile Bristeau, Rupert Klein, and Benoit Perthame (2004)
  A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows
  [DOI: 10.1137/S1064827503431090](https://doi.org/10.1137/S1064827503431090)
"""
@inline function hydrostatic_reconstruction_audusse_etal(u_ll, u_rr, equations::ShallowWaterEquations1D)
  # Unpack left and right water heights and bottom topographies
  h_ll, _, b_ll = u_ll
  h_rr, _, b_rr = u_rr

  # Get the velocities on either side
  v1_ll = velocity(u_ll, equations)
  v1_rr = velocity(u_rr, equations)

  # Compute the reconstructed water heights
  h_ll_star = max(zero(h_ll) , h_ll + b_ll - max(b_ll, b_rr) )
  h_rr_star = max(zero(h_rr) , h_rr + b_rr - max(b_ll, b_rr) )

  # Create the conservative variables using the reconstruted water heights
  u_ll_star = SVector( h_ll_star , h_ll_star * v1_ll , b_ll )
  u_rr_star = SVector( h_rr_star , h_rr_star * v1_rr , b_rr )

  return u_ll_star, u_rr_star
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::ShallowWaterEquations1D)
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
@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr, orientation_or_normal_direction,
                                                              equations::ShallowWaterEquations1D)
  λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction, equations)
  diss = -0.5 * λ * (u_rr - u_ll)
  return SVector(diss[1], diss[2], zero(eltype(u_ll)))
end


# Specialized `FluxHLL` to avoid spurious dissipation in the bottom topography
@inline function (numflux::FluxHLL)(u_ll, u_rr, orientation_or_normal_direction,
                                    equations::ShallowWaterEquations1D)
  λ_min, λ_max = numflux.min_max_speed(u_ll, u_rr, orientation_or_normal_direction, equations)

  if λ_min >= 0 && λ_max >= 0
    return flux(u_ll, orientation_or_normal_direction, equations)
  elseif λ_max <= 0 && λ_min <= 0
    return flux(u_rr, orientation_or_normal_direction, equations)
  else
    f_ll = flux(u_ll, orientation_or_normal_direction, equations)
    f_rr = flux(u_rr, orientation_or_normal_direction, equations)
    inv_λ_max_minus_λ_min = inv(λ_max - λ_min)
    factor_ll = λ_max * inv_λ_max_minus_λ_min
    factor_rr = λ_min * inv_λ_max_minus_λ_min
    factor_diss = λ_min * λ_max * inv_λ_max_minus_λ_min
    diss = u_rr - u_ll
    return factor_ll * f_ll - factor_rr * f_rr + factor_diss * SVector(diss[1], diss[2], zero(eltype(u_ll)))
  end
end


# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::ShallowWaterEquations1D)
  h_ll = waterheight(u_ll, equations)
  v_ll = velocity(u_ll, equations)
  h_rr = waterheight(u_rr, equations)
  v_rr = velocity(u_rr, equations)

  λ_min = v_ll - sqrt(equations.gravity * h_ll)
  λ_max = v_rr + sqrt(equations.gravity * h_rr)

  return λ_min, λ_max
end


@inline function max_abs_speeds(u, equations::ShallowWaterEquations1D)
  h = waterheight(u, equations)
  v = velocity(u, equations)

  c = equations.gravity * sqrt(h)
  return (abs(v) + c,)
end


# Helper function to extract the velocity vector from the conservative variables
@inline function velocity(u, equations::ShallowWaterEquations1D)
  h, h_v, _ = u

  v = h_v / h

  return v
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::ShallowWaterEquations1D)
  h, _, b = u

  H = h + b
  v = velocity(u, equations)
  return SVector(H, v, b)
end


# Convert conservative variables to entropy
# Note, only the first two are the entropy variables, the third entry still
# just carries the bottom topography values for convenience
@inline function cons2entropy(u, equations::ShallowWaterEquations1D)
  h, h_v, b = u

  v = velocity(u, equations)

  w1 = equations.gravity * (h + b) - 0.5 * v^2
  w2 = v

  return SVector(w1, w2, b)
end


# Convert entropy variables to conservative
@inline function entropy2cons(w, equations::ShallowWaterEquations1D)
  w1, w2, b = w

  h = (w1 + 0.5 * w2^2) / equations.gravity - b
  h_v = h * w2
  return SVector(h, h_v, b)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::ShallowWaterEquations1D)
  H, v, b = prim

  h = H - b
  h_v = h * v

  return SVector(h, h_v, b)
end


@inline function waterheight(u, equations::ShallowWaterEquations1D)
  return u[1]
end


@inline function pressure(u, equations::ShallowWaterEquations1D)
  h = waterheight(u, equations)
  p = 0.5 * equations.gravity * h^2
  return p
end


@inline function waterheight_pressure(u, equations::ShallowWaterEquations1D)
  return waterheight(u, equations) * pressure(u, equations)
end


# Entropy function for the shallow water equations is the total energy
@inline entropy(cons, equations::ShallowWaterEquations1D) = energy_total(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline function energy_total(cons, equations::ShallowWaterEquations1D)
  h, h_v, b = cons

  e = (h_v^2) / (2 * h) + 0.5 * equations.gravity * h^2 + equations.gravity * h * b
  return e
end


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::ShallowWaterEquations1D)
  h, h_v, _ = u
  return (h_v^2) / (2 * h)
end


# Calculate potential energy for a conservative state `cons`
@inline function energy_internal(cons, equations::ShallowWaterEquations1D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end


# Calculate the error for the "lake-at-rest" test case where H = h+b should
# be a constant value over time. Note, assumes there is a single reference
# water height `H0` with which to compare.
@inline function lake_at_rest_error(u, equations::ShallowWaterEquations1D)
  h, _, b = u

  #
  #  TODO: Compute this the proper way with
  #        H0_wet_dry = max( equations.H0 , b + equations.threshold_limiter )
  #        once PR for wet/dry `ShallowWaterEquations1D` is merged into this PR
  #

  return abs(equations.H0 - (h + b))
end


end # @muladd
