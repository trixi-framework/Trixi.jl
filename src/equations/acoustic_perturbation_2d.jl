@doc raw"""
    AcousticPerturbationEquations2D(v_mean_global, c_mean_global, rho_mean_global)

!!! warning "Experimental code"
    This system of equations is experimental and may change in any future release.

Acoustic perturbation equations (APE) in two space dimensions. The equations are given by
```math
\begin{aligned}
  \frac{\partial\mathbf{v'}}{\partial t} + \nabla (\bar{\mathbf{v}}\cdot\mathbf{v'})
    + \nabla\left( \frac{p'}{\bar{\rho}} \right) &= 0 \\
  \frac{\partial p'}{\partial t} + \nabla\cdot (\bar{c}^2 \bar{\rho}^2 \mathbf{v'} + \bar{v} p')
    &= \left( \bar{\rho}\mathbf{v'} + \bar{\mathbf{v}}\frac{p'}{\bar{c}^2} \right)\cdot\nabla\bar{c}^2.
\end{aligned}
```
The bar ``\bar{(\cdot)}`` indicates time-averaged quantities. The unknowns of the APE are the
perturbed velocities ``\mathbf{v'} = (v_1', v_2')^T`` and the perturbed pressure ``p'``, where
perturbed variables are defined by ``\phi' = \phi - \bar{\phi}``.

Note that the source term must be defined separately and passed manually to
[`SemidiscretizationHyperbolic`](@ref).

In addition to the unknowns, Trixi currently stores the mean values in the state vector,
i.e. the state vector used internally is given by
```math
\mathbf{u} =
  \begin{pmatrix}
    v_1' \\ v_2' \\ p' \\ \bar{v_1} \\ \bar{v_2} \\ \bar{c} \\ \bar{\rho}
  \end{pmatrix}.
```
This affects the implementation and use of these equations in various ways:
* The flux values corresponding to the mean values must be zero.
* The mean values have to be considered when defining initial conditions, boundary conditions or
  source terms.
* [`AnalysisCallback`](@ref) analyzes these variables too.
* Trixi's visualization tools will visualize the mean values by default.

The constructor accepts a 2-tuple `v_mean_global` and scalars `c_mean_global` and `rho_mean_global`
which can be used to make the definition of initial conditions for problems with constant mean flow
more flexible. These values are ignored if the mean values are defined internally in an initial
condition.

The equations are based on the APE-4 system introduced in the following paper:
- Roland Ewert and Wolfgang Schröder (2003)
  Acoustic perturbation equations based on flow decomposition via source filtering
  [DOI: 10.1016/S0021-9991(03)00168-2](https://doi.org/10.1016/S0021-9991(03)00168-2)
"""
struct AcousticPerturbationEquations2D{RealT<:Real} <: AbstractAcousticPerturbationEquations{2, 7}
  v_mean_global::SVector{2, RealT}
  c_mean_global::RealT
  rho_mean_global::RealT
end

function AcousticPerturbationEquations2D(v_mean_global::NTuple{2,<:Real}, c_mean_global::Real,
                                         rho_mean_global::Real)
  return AcousticPerturbationEquations2D(SVector(v_mean_global), c_mean_global, rho_mean_global)
end

function AcousticPerturbationEquations2D(; v_mean_global::NTuple{2,<:Real}, c_mean_global::Real,
                                         rho_mean_global::Real)
  return AcousticPerturbationEquations2D(SVector(v_mean_global), c_mean_global, rho_mean_global)
end


varnames(::typeof(cons2cons), ::AcousticPerturbationEquations2D) = ("v1_prime", "v2_prime", "p_prime",
                                                                    "v1_mean", "v2_mean", "c_mean", "rho_mean")
varnames(::typeof(cons2prim), ::AcousticPerturbationEquations2D) = ("v1_prime", "v2_prime", "p_prime",
                                                                    "v1_mean", "v2_mean", "c_mean", "rho_mean")


# Convenience functions for retrieving state variables and mean variables
function cons2state(u, equations::AcousticPerturbationEquations2D)
  return SVector(u[1], u[2], u[3])
end

function cons2mean(u, equations::AcousticPerturbationEquations2D)
  return SVector(u[4], u[5], u[6], u[7])
end

varnames(::typeof(cons2state), ::AcousticPerturbationEquations2D) = ("v1_prime", "v2_prime", "p_prime")
varnames(::typeof(cons2mean), ::AcousticPerturbationEquations2D) = ("v1_mean", "v2_mean", "c_mean", "rho_mean")


"""
    global_mean_vars(equations::AcousticPerturbationEquations2D)

Returns the global mean variables stored in `equations`. This makes it easier to define flexible
initial conditions for problems with constant mean flow.
"""
function global_mean_vars(equations::AcousticPerturbationEquations2D)
  return equations.v_mean_global[1], equations.v_mean_global[2], equations.c_mean_global,
         equations.rho_mean_global
end


"""
    initial_condition_constant(x, t, equations::AcousticPerturbationEquations2D)

A constant initial condition where the state variables are zero and the mean flow is constant.
Uses the global mean values from `equations`.
"""
function initial_condition_constant(x, t, equations::AcousticPerturbationEquations2D)
  v1_prime = 0.0
  v2_prime = 0.0
  p_prime = 0.0

  return SVector(v1_prime, v2_prime, p_prime, global_mean_vars(equations)...)
end


"""
    initial_condition_convergence_test(x, t, equations::AcousticPerturbationEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref). Uses the global mean values from `equations`.
"""
function initial_condition_convergence_test(x, t, equations::AcousticPerturbationEquations2D)
  c = 2.0
  A = 0.2
  L = 2.0
  f = 2.0 / L
  a = 1.0
  omega = 2 * pi * f
  init = c + A * sin(omega * (x[1] + x[2] - a*t))

  v1_prime = init
  v2_prime = init
  p_prime = init^2

  return SVector(v1_prime, v2_prime, p_prime, global_mean_vars(equations)...)
end

"""
  source_terms_convergence_test(u, x, t, equations::AcousticPerturbationEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref).
"""
function source_terms_convergence_test(u, x, t, equations::AcousticPerturbationEquations2D)
  v1_mean, v2_mean, c_mean, rho_mean = cons2mean(u, equations)

  c = 2.0
  A = 0.2
  L = 2.0
  f = 2.0 / L
  a = 1.0
  omega = 2 * pi * f

  si, co = sincos(omega * (x[1] + x[2] - a * t))
  tmp = v1_mean + v2_mean - a

  du1 = du2 = A * omega * co * (2 * c + tmp + 2/rho_mean * A * si)
  du3 = A * omega * co * (2 * c_mean^2 * rho_mean + 2 * c * tmp + 2 * A * tmp * si)

  du4 = du5 = du6 = du7 = 0.0

  return SVector(du1, du2, du3, du4, du5, du6, du7)
end


"""
    initial_condition_gauss(x, t, equations::AcousticPerturbationEquations2D)

A Gaussian pulse in a constant mean flow. Uses the global mean values from `equations`.
"""
function initial_condition_gauss(x, t, equations::AcousticPerturbationEquations2D)
  v1_prime = 0.0
  v2_prime = 0.0
  p_prime = exp(-4*(x[1]^2 + x[2]^2))

  return SVector(v1_prime, v2_prime, p_prime, global_mean_vars(equations)...)
end


"""
    initial_condition_gauss_wall(x, t, equations::AcousticPerturbationEquations2D)

A Gaussian pulse, used in the `gauss_wall` example elixir in combination with
[`boundary_condition_wall`](@ref). Uses the global mean values from `equations`.
"""
function initial_condition_gauss_wall(x, t, equations::AcousticPerturbationEquations2D)
  v1_prime = 0.0
  v2_prime = 0.0
  p_prime = exp(-log(2) * (x[1]^2 + (x[2] - 25)^2) / 25)

  return SVector(v1_prime, v2_prime, p_prime, global_mean_vars(equations)...)
end

"""
    boundary_condition_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                            equations::AcousticPerturbationEquations2D)

Boundary conditions for a solid wall.
"""
function boundary_condition_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                                 equations::AcousticPerturbationEquations2D)
  # Boundary state is equal to the inner state except for the perturbed velocity. For boundaries
  # in the -x/+x direction, we multiply the perturbed velocity in the x direction by -1.
  # Similarly, for boundaries in the -y/+y direction, we multiply the perturbed velocity in the
  # y direction by -1
  if direction in (1, 2) # x direction
    u_boundary = SVector(-u_inner[1], u_inner[2], u_inner[3], cons2mean(u_inner, equations)...)
  else # y direction
    u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], cons2mean(u_inner, equations)...)
  end

  # Calculate boundary flux
  if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


"""
  initial_condition_monopole(x, t, equations::AcousticPerturbationEquations2D)

Initial condition for the monopole in a boundary layer setup, used in combination with
[`boundary_condition_monopole`](@ref).
"""
function initial_condition_monopole(x, t, equations::AcousticPerturbationEquations2D)
  m = 0.3 # Mach number

  v1_prime = 0.0
  v2_prime = 0.0
  p_prime = 0.0

  v1_mean = x[2] > 1 ? m : m * (2*x[2] - 2*x[2]^2 + x[2]^4)
  v2_mean = 0.0
  c_mean = 1.0
  rho_mean = 1.0

  return SVector(v1_prime, v2_prime, p_prime, v1_mean, v2_mean, c_mean, rho_mean)
end

"""
  boundary_condition_monopole(u_inner, orientation, direction, x, t, surface_flux_function,
                              equations::AcousticPerturbationEquations2D)

Boundary condition for a monopole in a boundary layer at the -y boundary, i.e. `direction = 3`.
This will return an error for any other direction. This boundary condition is used in combination
with [`initial_condition_monopole`](@ref).
"""
function boundary_condition_monopole(u_inner, orientation, direction, x, t, surface_flux_function,
                                     equations::AcousticPerturbationEquations2D)
  if direction != 3
    error("expected direction = 3, got $direction instead")
  end

  # Wall at the boundary in -y direction with a monopole at -0.05 <= x <= 0.05. In the monopole area
  # we use a sinusoidal boundary state for the perturbed variables. For the rest of the -y boundary
  # we set the boundary state to the inner state and multiply the perturbed velocity in the
  # y-direction by -1.
  if -0.05 <= x[1] <= 0.05 # Monopole
    v1_prime = 0.0
    v2_prime = p_prime = sin(2 * pi * t)

    u_boundary = SVector(v1_prime, v2_prime, p_prime, u_inner[4], u_inner[5], u_inner[6],
                         u_inner[7])
  else # Wall
    u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4], u_inner[5], u_inner[6],
                         u_inner[7])
  end

  # Calculate boundary flux
  flux = surface_flux_function(u_boundary, u_inner, orientation, equations)

  return flux
end

"""
    boundary_condition_zero(u_inner, orientation, direction, x, t, surface_flux_function,
                            equations::AcousticPerturbationEquations2D)

Boundary condition that uses a boundary state where the state variables are zero and the mean
variables are the same as in `u_inner`.
"""
function boundary_condition_zero(u_inner, orientation, direction, x, t, surface_flux_function,
                                 equations::AcousticPerturbationEquations2D)
  value = zero(eltype(u_inner))
  u_boundary = SVector(value, value, value, cons2mean(u_inner, equations)...)

  # Calculate boundary flux
  if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::AcousticPerturbationEquations2D)
  v1_prime, v2_prime, p_prime = cons2state(u, equations)
  v1_mean, v2_mean, c_mean, rho_mean = cons2mean(u, equations)

  # Calculate flux for conservative state variables
  if orientation == 1
    f1 = v1_mean * v1_prime + v2_mean * v2_prime + p_prime / rho_mean
    f2 = zero(eltype(u))
    f3 = c_mean^2 * rho_mean * v1_prime + v1_mean * p_prime
  else
    f1 = zero(eltype(u))
    f2 = v1_mean * v1_prime + v2_mean * v2_prime + p_prime / rho_mean
    f3 = c_mean^2 * rho_mean * v2_prime + v2_mean * p_prime
  end

  # The rest of the state variables are actually variable coefficients, hence the flux should be
  # zero. See https://github.com/trixi-framework/Trixi.jl/issues/358#issuecomment-784828762
  # for details.
  f4 = f5 = f6 = f7 = zero(eltype(u))

  return SVector(f1, f2, f3, f4, f5, f6, f7)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::AcousticPerturbationEquations2D)
  # Calculate v = v_prime + v_mean
  v_prime_ll = u_ll[orientation]
  v_prime_rr = u_rr[orientation]
  v_mean_ll = u_ll[orientation + 3]
  v_mean_rr = u_rr[orientation + 3]

  v_ll = v_prime_ll + v_mean_ll
  v_rr = v_prime_rr + v_mean_rr

  c_mean_ll = u_ll[6]
  c_mean_rr = u_rr[6]

  λ_max = max(abs(v_ll), abs(v_rr)) + max(c_mean_ll, c_mean_rr)
end


# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector, equations::AcousticPerturbationEquations2D)
  v1_prime, v2_prime, p_prime = cons2state(u, equations)
  v1_mean, v2_mean, c_mean, rho_mean = cons2mean(u, equations)

  f1 = normal_direction[1] * (v1_mean * v1_prime + v2_mean * v2_prime + p_prime / rho_mean)
  f2 = normal_direction[2] * (v1_mean * v1_prime + v2_mean * v2_prime + p_prime / rho_mean)
  f3 = ( normal_direction[1] * (c_mean^2 * rho_mean * v1_prime + v1_mean * p_prime)
       + normal_direction[2] * (c_mean^2 * rho_mean * v2_prime + v2_mean * p_prime) )

  # The rest of the state variables are actually variable coefficients, hence the flux should be
  # zero. See https://github.com/trixi-framework/Trixi.jl/issues/358#issuecomment-784828762
  # for details.
  f4 = f5 = f6 = f7 = zero(eltype(u))

  return SVector(f1, f2, f3, f4, f5, f6, f7)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::AcousticPerturbationEquations2D)
  # Calculate v = v_prime + v_mean
  v_prime_ll = normal_direction[1]*u_ll[1] + normal_direction[2]*u_ll[2]
  v_prime_rr = normal_direction[1]*u_rr[1] + normal_direction[2]*u_rr[2]
  v_mean_ll = normal_direction[1]*u_ll[4] + normal_direction[2]*u_ll[5]
  v_mean_rr = normal_direction[1]*u_rr[4] + normal_direction[2]*u_rr[5]

  v_ll = v_prime_ll + v_mean_ll
  v_rr = v_prime_rr + v_mean_rr

  c_mean_ll = u_ll[6]
  c_mean_rr = u_rr[6]

  λ_max = (max(abs(v_ll), abs(v_rr)) + max(c_mean_ll, c_mean_rr)) * norm(normal_direction)
end


"""
    boundary_state_slip_wall(u_inner, normal_direction::AbstractVector,
                             equations::AcousticPertubationEquations2D)

Idea behind this boundary condition is to use an orthogonal projection of the perturbed velocities
to zero out the normal velocity while retaining the possibility of a tangential velocity
in the boundary state. Further details are available in the paper:
- Marcus Bauer, Jürgen Dierke and Roland Ewert (2011)
  Application of a discontinuous Galerkin method to discretize acoustic perturbation equations
  [DOI: 10.2514/1.J050333](https://doi.org/10.2514/1.J050333)
"""
function boundary_state_slip_wall(u_inner, normal_direction::AbstractVector,
                                  equations::AcousticPerturbationEquations2D)
  # normalize the outward pointing direction
  normal = normal_direction / norm(normal_direction)

  # compute the normal and tangential components of the velocity
  u_normal  = normal[1] * u_inner[1] + normal[2] * u_inner[2]
  u_tangent = (u_inner[1] - u_normal * normal[1], u_inner[2] - u_normal * normal[2])

  return SVector(u_tangent[1] - u_normal * normal[1],
                 u_tangent[2] - u_normal * normal[2],
                 u_inner[3],
                 cons2mean(u_inner, equations)...)
end



@inline have_constant_speed(::AcousticPerturbationEquations2D) = Val(false)

@inline function max_abs_speeds(u, equations::AcousticPerturbationEquations2D)
  v1_mean = u[4]
  v2_mean = u[5]
  c_mean = u[6]

  return abs(v1_mean) + c_mean, abs(v2_mean) + c_mean
end


# Convert conservative variables to primitive
@inline cons2prim(u, equations::AcousticPerturbationEquations2D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equations::AcousticPerturbationEquations2D) = u
