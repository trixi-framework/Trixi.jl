@doc raw"""
    AcousticPerturbationEquations2D(v_mean, rho_mean, c_mean)

Acoustic perturbation equations in two space dimensions with constant mean flow. The equations are
based on the APE-4 system introduced by Ewert and Schröder
[DOI: 10.1016/S0021-9991(03)00168-2](https://doi.org/10.1016/S0021-9991(03)00168-2).
"""
struct AcousticPerturbationEquations2D{RealT<:Real} <: AbstractAcousticPerturbationEquations{2, 3}
  v_mean::SVector{2, RealT}
  rho_mean::RealT
  c_mean::RealT
end

function AcousticPerturbationEquations2D(v_mean::NTuple{2,<:Real}, rho_mean::Real, c_mean::Real)
  return AcousticPerturbationEquations2D(SVector(v_mean), rho_mean, c_mean)
end


get_name(::AcousticPerturbationEquations2D) = "AcousticPerturbationEquations2D"
varnames(::typeof(cons2cons), ::AcousticPerturbationEquations2D) = ("v1_prime", "v2_prime", "p_prime")
varnames(::typeof(cons2prim), ::AcousticPerturbationEquations2D) = ("v1_prime", "v2_prime", "p_prime")


"""
    initial_condition_constant(x, t, equations::AcousticPerturbationEquations2D)

A constant initial condition.
"""
function initial_condition_constant(x, t, equations::AcousticPerturbationEquations2D)
  v1_prime = 0.0
  v2_prime = 0.0
  p_prime = 0.0

  return SVector(v1_prime, v2_prime, p_prime)
end


"""
initial_condition_gauss(x, t, equation::AcousticPerturbationEquations2D)

A Gaussian pulse.
"""
function initial_condition_gauss(x, t, equations::AcousticPerturbationEquations2D)
  v1_prime = 0.0
  v2_prime = 0.0
  p_prime = exp(-4*(x[1]^2 + x[2]^2))

  return SVector(v1_prime, v2_prime, p_prime)
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
  p = init * init

  return SVector(v1_prime, v2_prime, p)
end

"""
  source_terms_convergence_test(u, x, t, equations::AcousticPerturbationEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref).
"""
function source_terms_convergence_test(u, x, t, equations::AcousticPerturbationEquations2D)
  @unpack v_mean, rho_mean, c_mean = equations

  c = 2.0
  A = 0.2
  L = 2.0
  f = 2.0 / L
  a = 1.0
  omega = 2 * pi * f

  si, co = sincos(omega * (x[1] + x[2] - a * t))
  tmp = v_mean[1] + v_mean[2] - a

  du1 = du2 = A * omega * co * (2 * c + tmp + 2/rho_mean * A * si)
  du3 = A * omega * co * (2 * c_mean^2 * rho_mean + 2 * c * tmp + 2 * A * tmp * si)

  return SVector(du1, du2, du3)
end


# Calculate 1D flux for a single point
@inline function calcflux(u, orientation, equations::AcousticPerturbationEquations2D)
  v1_prime, v2_prime, p_prime = u
  @unpack v_mean, rho_mean, c_mean = equations

  if orientation == 1
    f1 = v_mean[1] * v1_prime + v_mean[2] * v2_prime + p_prime / rho_mean
    f2 = zero(eltype(u))
    f3 = c_mean^2 * rho_mean * v1_prime + v_mean[orientation] * p_prime
  else
    f1 = zero(eltype(u))
    f2 = v_mean[1] * v1_prime + v_mean[2] * v2_prime + p_prime / rho_mean
    f3 = c_mean^2 * rho_mean * v2_prime + v_mean[orientation] * p_prime
  end

  return SVector(f1, f2, f3)
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equations::AcousticPerturbationEquations2D)
  f_ll = calcflux(u_ll, orientation, equations)
  f_rr = calcflux(u_rr, orientation, equations)

  v_ll = u_ll[orientation] + equations.v_mean[orientation]
  v_rr = u_rr[orientation] + equations.v_mean[orientation]
  c0 = equations.c_mean
  speed = max(abs(v_ll), abs(v_rr)) + c0

  return 0.5 * ( (f_ll + f_rr) - speed * (u_rr - u_ll) )
end


@inline have_constant_speed(::AcousticPerturbationEquations2D) = Val(true)

@inline function max_abs_speeds(equations::AcousticPerturbationEquations2D)
  return abs.(equations.v_mean) .+ equations.c_mean
end


# Convert conservative variables to primitive
@inline cons2prim(u, equations::AcousticPerturbationEquations2D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equations::AcousticPerturbationEquations2D) = u


# Calculate entropy for a conservative state `cons`
@inline entropy(u::Real, ::AcousticPerturbationEquations2D) = 0.5 * u^2
@inline entropy(u, equations::AcousticPerturbationEquations2D) = entropy(u[1], equations)


# Calculate total energy for a conservative state `cons`
@inline energy_total(u::Real, ::AcousticPerturbationEquations2D) = 0.5 * u^2
@inline energy_total(u, equations::AcousticPerturbationEquations2D) = energy_total(u[1], equations)
