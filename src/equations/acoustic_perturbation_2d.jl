@doc raw"""
    AcousticPerturbationEquations2D()

!!! warning "Experimental code"
    This system of equations is experimental and may change in any future release.

Acoustic perturbation equations in two space dimensions.

The equations are based on the APE-4 system introduced in the following paper:

R. Ewert, W. Schröder
"Acoustic perturbation equations based on flow decomposition via source filtering",
Journal of Computational Physics,
Volume 188, Issue 2,
2003,
[DOI: 10.1016/S0021-9991(03)00168-2](https://doi.org/10.1016/S0021-9991(03)00168-2)
"""
struct AcousticPerturbationEquations2D <: AbstractAcousticPerturbationEquations{2, 7} end


get_name(::AcousticPerturbationEquations2D) = "AcousticPerturbationEquations2D"
varnames(::typeof(cons2cons), ::AcousticPerturbationEquations2D) = ("v1_prime", "v2_prime", "p_prime",
                                                                    "v1_mean", "v2_mean", "c_mean", "rho_mean")
varnames(::typeof(cons2prim), ::AcousticPerturbationEquations2D) = ("v1_prime", "v2_prime", "p_prime",
                                                                    "v1_mean", "v2_mean", "c_mean", "rho_mean")


"""
    initial_condition_constant(x, t, equations::AcousticPerturbationEquations2D)

A constant initial condition.
"""
function initial_condition_constant(x, t, equations::AcousticPerturbationEquations2D)
  v1_prime = 0.0
  v2_prime = 0.0
  p_prime = 0.0

  v1_mean = -0.5
  v2_mean = 0.25
  c_mean = 1.0
  rho_mean = 1.0

  return SVector(v1_prime, v2_prime, p_prime, v1_mean, v2_mean, c_mean, rho_mean)
end


"""
    initial_condition_convergence_test(x, t, equations::AcousticPerturbationEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref).
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
  p = init^2

  v1_mean = 0.5
  v2_mean = 0.3
  c_mean = 1.0
  rho_mean = 1.0

  return SVector(v1_prime, v2_prime, p, v1_mean, v2_mean, c_mean, rho_mean)
end

"""
  source_terms_convergence_test(u, x, t, equations::AcousticPerturbationEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref).
"""
function source_terms_convergence_test(u, x, t, equations::AcousticPerturbationEquations2D)
  v1_mean = u[4]
  v2_mean = u[5]
  c_mean = u[6]
  rho_mean = u[7]

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

A Gaussian pulse.
"""
function initial_condition_gauss(x, t, equations::AcousticPerturbationEquations2D)
  v1_prime = 0.0
  v2_prime = 0.0
  p_prime = exp(-4*(x[1]^2 + x[2]^2))

  v1_mean = 0.25
  v2_mean = 0.25
  c_mean = 1.0
  rho_mean = 1.0

  return SVector(v1_prime, v2_prime, p_prime, v1_mean, v2_mean, c_mean, rho_mean)
end


"""
    initial_condition_gauss_wall(x, t, equations::AcousticPerturbationEquations2D)

A Gaussian pulse, used in the `gauss_wall` example elixir in combination with
[`boundary_condition_gauss_wall`](@ref).
"""
function initial_condition_gauss_wall(x, t, equations::AcousticPerturbationEquations2D)
  v1_prime = 0.0
  v2_prime = 0.0
  p_prime = exp(-log(2) * (x[1]^2 + (x[2] - 25)^2) / 25)

  v1_mean = 0.5
  v2_mean = 0.0
  c_mean = 1.0
  rho_mean = 1.0

  return SVector(v1_prime, v2_prime, p_prime, v1_mean, v2_mean, c_mean, rho_mean)
end

"""
    boundary_condition_gauss_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                                  equations::AcousticPerturbationEquations2D)

Boundary condition for the `gauss_wall` example elixir, used in combination with
[`initial_condition_gauss_wall`](@ref).
"""
function boundary_condition_gauss_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                                       equations::AcousticPerturbationEquations2D)
  # Calculate boundary flux
  if direction == 1 # Boundary at -x
    u_boundary = SVector(0.0, 0.0, 0.0, u_inner[4], u_inner[5], u_inner[6], u_inner[7])
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  elseif direction == 2 # Boundary at +x
    u_boundary = SVector(0.0, 0.0, 0.0, u_inner[4], u_inner[5], u_inner[6], u_inner[7])
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  elseif direction == 3 # Boundary at -y
    u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4], u_inner[5], u_inner[6],
                         u_inner[7])
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  else # Boundary at +y
    u_boundary = SVector(0.0, 0.0, 0.0, u_inner[4], u_inner[5], u_inner[6], u_inner[7])
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
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

Boundary condition for the monopole in a boundary layer setup, used in combination with
[`initial_condition_monopole`](@ref).
"""
function boundary_condition_monopole(u_inner, orientation, direction, x, t, surface_flux_function,
                                     equations::AcousticPerturbationEquations2D)
  # Calculate boundary flux
  if direction == 1 # Boundary at -x
    u_boundary = SVector(0.0, 0.0, 0.0, u_inner[4], u_inner[5], u_inner[6], u_inner[7])
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  elseif direction == 2 # Boundary at +x
    u_boundary = SVector(0.0, 0.0, 0.0, u_inner[4], u_inner[5], u_inner[6], u_inner[7])
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  elseif direction == 3 # Boundary at -y
    if -0.05 <= x[1] <= 0.05 # Monopole
      v1_prime = 0.0
      v2_prime = p_prime = sin(2 * pi * t)

      u_boundary = SVector(v1_prime, v2_prime, p_prime, u_inner[4], u_inner[5], u_inner[6],
                           u_inner[7])
    else # Wall
      u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4], u_inner[5], u_inner[6],
                           u_inner[7])
    end

    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  else # Boundary at +y
    u_boundary = SVector(0.0, 0.0, 0.0, u_inner[4], u_inner[5], u_inner[6], u_inner[7])
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  end

  return flux
end


# Calculate 1D flux for a single point
@inline function calcflux(u, orientation, equations::AcousticPerturbationEquations2D)
  v1_prime, v2_prime, p_prime, v1_mean, v2_mean, c_mean, rho_mean = u

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


function flux_lax_friedrichs(u_ll, u_rr, orientation, equations::AcousticPerturbationEquations2D)
  f_ll = calcflux(u_ll, orientation, equations)
  f_rr = calcflux(u_rr, orientation, equations)

  # Calculate v = v_prime + v_mean
  v_ll = u_ll[orientation] + u_ll[orientation + 3]
  v_rr = u_rr[orientation] + u_rr[orientation + 3]

  c_mean_ll = u_ll[6]
  c_mean_rr = u_rr[6]

  speed = max(abs(v_ll), abs(v_rr)) + max(c_mean_ll, c_mean_rr)

  return 0.5 * ( (f_ll + f_rr) - speed * (u_rr - u_ll) )
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