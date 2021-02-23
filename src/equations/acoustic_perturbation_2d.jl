@doc raw"""
TODO
"""
struct AcousticPerturbationEquations2D{RealT<:Real} <: AbstractAcousticPerturbationEquations{2, 3}
  v_avg::SVector{2, RealT}
  rho_avg::RealT
  c_sq_avg::RealT
end

function AcousticPerturbationEquations2D(v_avg::NTuple{2,<:Real}, rho_avg::Real, c_sq_avg::Real)
  AcousticPerturbationEquations2D(SVector(v_avg), rho_avg, c_sq_avg)
end


get_name(::AcousticPerturbationEquations2D) = "AcousticPerturbationEquations2D"
varnames(::typeof(cons2cons), ::AcousticPerturbationEquations2D) = ("v1_prime", "v2_prime", "p_prime")
varnames(::typeof(cons2prim), ::AcousticPerturbationEquations2D) = ("v1_prime", "v2_prime", "p_prime")


"""
initial_condition_gauss(x, t, equation::AcousticPerturbationEquations2D)

A Gaussian pulse.
"""
function initial_condition_gauss(x, t, equations::AcousticPerturbationEquations2D)
  p_prime = exp(-4*(x[1]^2 + x[2]^2))
  v1_prime = 0.0
  v2_prime = 0.0

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
  @unpack v_avg, rho_avg, c_sq_avg = equations

  c = 2.0
  A = 0.2
  L = 2.0
  f = 2.0 / L
  a = 1.0
  omega = 2 * pi * f

  si, co = sincos(omega * (x[1] + x[2] - a * t))
  tmp = v_avg[1] + v_avg[2] - a

  du1 = du2 = A * omega * co * (2 * c + tmp + 2/rho_avg * A * si)
  du3 = A * omega * co * (2 * c_sq_avg * rho_avg + 2 * c * tmp + 2 * A * tmp * si)

  return SVector(du1, du2, du3)
end


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

# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equations::AcousticPerturbationEquations2D)
  v1_prime, v2_prime, p_prime = u
  @unpack v_avg, rho_avg, c_sq_avg = equations

  if orientation == 1
    f1 = v_avg[1] * v1_prime + v_avg[2] * v2_prime + p_prime / rho_avg
    f2 = zero(eltype(u))
    f3 = c_sq_avg * rho_avg * v1_prime + v_avg[orientation] * p_prime
  else
    f1 = zero(eltype(u))
    f2 = v_avg[1] * v1_prime + v_avg[2] * v2_prime + p_prime / rho_avg
    f3 = c_sq_avg * rho_avg * v2_prime + v_avg[orientation] * p_prime
  end

  return SVector(f1, f2, f3)
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equations::AcousticPerturbationEquations2D)
  f_ll = calcflux(u_ll, orientation, equations)
  f_rr = calcflux(u_rr, orientation, equations)

  v_ll = u_ll[orientation] + equations.v_avg[orientation]
  v_rr = u_rr[orientation] + equations.v_avg[orientation]
  c0 = sqrt(equations.c_sq_avg)
  speed = max(abs(v_ll), abs(v_rr)) + c0

  return 0.5 * ( (f_ll + f_rr) - speed * (u_rr - u_ll) )
end


@inline have_constant_speed(::AcousticPerturbationEquations2D) = Val(true)

@inline function max_abs_speeds(equations::AcousticPerturbationEquations2D)
  return abs.(equations.v_avg) .+ sqrt(equations.c_sq_avg)
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
