
@doc raw"""
    LatticeBoltzmannEquation2D

The Lattice-Boltzmann equation
```math
\partial_t f_\alpha + v_{\alpha,1} \partial_1 f_\alpha + v_{\alpha,2} \partial_2 f_\alpha = 0
```
in two space dimensions.
"""
struct LatticeBoltzmannEquation2D{RealT<:Real} <: AbstractLatticeBoltzmannEquation{2, 9}
  c_s::RealT
  c::RealT
  v_alpha1::SVector{9, RealT}
  v_alpha2::SVector{9, RealT}
  # v_alpha::SMatrix{(9,2), RealT, 2, 18}
end

function LatticeBoltzmannEquation2D(c_s::Real)
  c = convert(Real, sqrt(3)) * c_s
  v_alpha1 = @SVector ([ 0,  1,  0, -1,  0,  1, -1,  -1,  1] * c)
  v_alpha2 = @SVector ([ 0,  0,  1,  0, -1,  1,  1,  -1, -1] * c)
  # v_alpha = @SMatrix [ 0 0; c 0; 0 c; -c 0; 0 -c; c c; -c c; -c -c; c -c]
  LatticeBoltzmannEquation2D(c_s, c, v_alpha1, v_alpha2)
  # LatticeBoltzmannEquation2D(c_s, c, v_alpha)
end


get_name(::LatticeBoltzmannEquation2D) = "LatticeBoltzmannEquation2D"
varnames_cons(::LatticeBoltzmannEquation2D) = @SVector ["pdf"*string(i) for i in 1:9]
varnames_prim(::LatticeBoltzmannEquation2D) = @SVector ["rho", "v1", "v2", "p"]

# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::LatticeBoltzmannEquation2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equation::LatticeBoltzmannEquation2D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x_trans_periodic_2d(x - equation.advectionvelocity * t)

  return @SVector [2.0]
end


"""
    initial_condition_convergence_test(x, t, equations::LatticeBoltzmannEquation2D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equation::LatticeBoltzmannEquation2D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  c = 1.0
  A = 0.5
  L = 2
  f = 1/L
  omega = 2 * pi * f
  scalar = c + A * sin(omega * sum(x_trans))
  return @SVector [scalar]
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::LatticeBoltzmannEquation2D)


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equation::LatticeBoltzmannEquation2D)
  if orientation == 1
    v_alpha = equation.v_alpha1
  else
    v_alpha = equation.v_alpha2
  end
  return v_alpha .* u
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::LatticeBoltzmannEquation2D)
  if orientation == 1
    v_alpha = equation.v_alpha1
  else
    v_alpha = equation.v_alpha2
  end
  return 0.5 * ( v_alpha .* (u_ll + u_rr) - abs.(v_alpha) .* (u_rr - u_ll) )
end



@inline have_constant_speed(::LatticeBoltzmannEquation2D) = Val(true)

@inline function max_abs_speeds(equation::LatticeBoltzmannEquation2D)
  return SVector(1, 1) * equation.c
end


# Convert conservative variables to primitive
@inline cons2prim(u, equation::LatticeBoltzmannEquation2D) = error("not implemented")

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equation::LatticeBoltzmannEquation2D) = error("not implemented")


# Calculate entropy for a conservative state `cons`
@inline entropy(u, equation::LatticeBoltzmannEquation2D) = error("not implemented") 


# Calculate total energy for a conservative state `cons`
@inline energy_total(u, equation::LatticeBoltzmannEquation2D) = error("not implemented")
