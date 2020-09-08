
@doc raw"""
    HyperbolicDiffusionEquations2D

The linear hyperbolic diffusion equations in two space dimensions.
A description of this system can be found in Sec. 2.5 of the book "I Do Like CFD, Too: Vol 1".
The book is freely available at http://www.cfdbooks.com/ and further analysis can be found in
the paper by Nishikawa [DOI: 10.1016/j.jcp.2007.07.029](https://doi.org/10.1016/j.jcp.2007.07.029)
"""
struct HyperbolicDiffusionEquations2D <: AbstractHyperbolicDiffusionEquations{2, 3}
  Lr::Float64
  Tr::Float64
  nu::Float64
  resid_tol::Float64
end

function HyperbolicDiffusionEquations2D()
  # diffusion coefficient
  nu = parameter("nu", 1.0)
  # relaxation length scale
  Lr = parameter("Lr", 1.0/(2.0*pi))
  # relaxation time
  Tr = Lr*Lr/nu
  # stopping tolerance for the pseudotime "steady-state"
  resid_tol = parameter("resid_tol", 1e-12)
  HyperbolicDiffusionEquations2D(Lr, Tr, nu, resid_tol)
end


get_name(::HyperbolicDiffusionEquations2D) = "HyperbolicDiffusionEquations2D"
varnames_cons(::HyperbolicDiffusionEquations2D) = @SVector ["phi", "q1", "q2"]
varnames_prim(::HyperbolicDiffusionEquations2D) = @SVector ["phi", "q1", "q2"]
default_analysis_quantities(::HyperbolicDiffusionEquations2D) = (:l2_error, :linf_error, :residual)


# Set initial conditions at physical location `x` for pseudo-time `t`
function initial_conditions_poisson_periodic(x, t, equation::HyperbolicDiffusionEquations2D)
  # elliptic equation: -νΔϕ = f
  # depending on initial constant state, c, for phi this converges to the solution ϕ + c
  if iszero(t)
    phi = 0.0
    q1  = 0.0
    q2  = 0.0
  else
    phi = sin(2.0*pi*x[1])*sin(2.0*pi*x[2])
    q1  = 2*pi*cos(2.0*pi*x[1])*sin(2.0*pi*x[2])
    q2  = 2*pi*sin(2.0*pi*x[1])*cos(2.0*pi*x[2])
  end
  return @SVector [phi, q1, q2]
end

function initial_conditions_poisson_nonperiodic(x, t, equation::HyperbolicDiffusionEquations2D)
  # elliptic equation: -νΔϕ = f
  if t == 0.0
    phi = 1.0
    q1  = 1.0
    q2  = 1.0
  else
    phi = 2.0*cos(pi*x[1])*sin(2.0*pi*x[2]) + 2.0 # ϕ
    q1  = -2.0*pi*sin(pi*x[1])*sin(2.0*pi*x[2])   # ϕ_x
    q2  = 4.0*pi*cos(pi*x[1])*cos(2.0*pi*x[2])    # ϕ_y
  end
  return @SVector [phi, q1, q2]
end

function initial_conditions_harmonic_nonperiodic(x, t, equation::HyperbolicDiffusionEquations2D)
  # elliptic equation: -νΔϕ = f
  if t == 0.0
    phi = 1.0
    q1  = 1.0
    q2  = 1.0
  else
    C   = 1.0/sinh(pi)
    phi = C*(sinh(pi*x[1])*sin(pi*x[2]) + sinh(pi*x[2])*sin(pi*x[1]))
    q1  = C*pi*(cosh(pi*x[1])*sin(pi*x[2]) + sinh(pi*x[2])*cos(pi*x[1]))
    q2  = C*pi*(sinh(pi*x[1])*cos(pi*x[2]) + cosh(pi*x[2])*sin(pi*x[1]))
  end
  return @SVector [phi, q1, q2]
end

function initial_conditions_jeans_instability(x, t, equation::HyperbolicDiffusionEquations2D)
  # gravity equation: -Δϕ = -4πGρ
  # Constants taken from the FLASH manual
  # https://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel.pdf
  rho0 = 1.5e7
  delta0 = 1e-3
  #
  phi = rho0*delta0 # constant background pertubation magnitude
  q1  = 0.0
  q2  = 0.0
  return @SVector [phi, q1, q2]
end

function initial_conditions_eoc_test_coupled_euler_gravity(x, t, equation::HyperbolicDiffusionEquations2D)

  # Determine phi_x, phi_y
  G = 1.0 # gravitational constant
  C = -2.0*G/pi
  A = 0.1 # perturbation coefficient must match Euler setup
  rho1 = A * sin(pi * (x[1] + x[2] - t))
  # intialize with ansatz of gravity potential
  phi = C * rho1
  q1  = C * A * pi * cos(pi*(x[1] + x[2] - t)) # = gravity acceleration in x-direction
  q2  = q1                                     # = gravity acceleration in y-direction

  return @SVector [phi, q1, q2]
end


function initial_conditions_sedov_self_gravity(x, t, equation::HyperbolicDiffusionEquations2D)
  # for now just use constant initial condition for sedov blast wave (can likely be improved)
  phi = 0.0
  q1  = 0.0
  q2  = 0.0
  return @SVector [phi, q1, q2]
end

# Apply source terms
function source_terms_poisson_periodic(ut, u, x, element_id, t, n_nodes, equation::HyperbolicDiffusionEquations2D)
  # elliptic equation: -νΔϕ = f
  # analytical solution: phi = sin(2πx)*sin(2πy) and f = -8νπ^2 sin(2πx)*sin(2πy)
  inv_Tr = inv(equation.Tr)
  C = -8.0*equation.nu*pi*pi

  for j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, element_id]
    x2 = x[2, i, j, element_id]
    tmp1 = sin(2.0*pi*x1)
    tmp2 = sin(2.0*pi*x2)
    ut[1, i, j, element_id] -= C*tmp1*tmp2
    ut[2, i, j, element_id] -= inv_Tr * u[2, i, j, element_id]
    ut[3, i, j, element_id] -= inv_Tr * u[3, i, j, element_id]
  end

  return nothing
end

function source_terms_poisson_nonperiodic(ut, u, x, element_id, t, n_nodes, equation::HyperbolicDiffusionEquations2D)
  # elliptic equation: -νΔϕ = f
  # analytical solution: ϕ = 2cos(πx)sin(2πy) + 2 and f = 10π^2cos(πx)sin(2πy)
  inv_Tr = inv(equation.Tr)

  for j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, element_id]
    x2 = x[2, i, j, element_id]
    ut[1, i, j, element_id] += 10 * pi^2 * cos(pi*x1) * sin(2.0*pi*x2)
    ut[2, i, j, element_id] -= inv_Tr * u[2, i, j, element_id]
    ut[3, i, j, element_id] -= inv_Tr * u[3, i, j, element_id]
  end

  return nothing
end

function source_terms_harmonic(ut, u, x, element_id, t, n_nodes, equation::HyperbolicDiffusionEquations2D)
  # harmonic solution ϕ = (sinh(πx)sin(πy) + sinh(πy)sin(πx))/sinh(π), so f = 0
  inv_Tr = inv(equation.Tr)

  for j in 1:n_nodes, i in 1:n_nodes
    ut[2, i, j, element_id] -= inv_Tr * u[2, i, j, element_id]
    ut[3, i, j, element_id] -= inv_Tr * u[3, i, j, element_id]
  end

  return nothing
end

# The coupled EOC test does not require additional sources
function source_terms_eoc_test_coupled_euler_gravity(ut, u, x, element_id, t, n_nodes, equation::HyperbolicDiffusionEquations2D)
  return source_terms_harmonic(ut, u, x, element_id, t, n_nodes, equation)
end


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equation::HyperbolicDiffusionEquations2D)
  phi, q1, q2 = u

  if orientation == 1
    f1 = -equation.nu*q1
    f2 = -phi/equation.Tr
    f3 = zero(phi)
  else
    f1 = -equation.nu*q2
    f2 = zero(phi)
    f3 = -phi/equation.Tr
  end

  return SVector(f1, f2, f3)
end


@inline function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::HyperbolicDiffusionEquations2D)
  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  λ_max = sqrt(equation.nu / equation.Tr)

  return 0.5 * (f_ll + f_rr - λ_max * (u_rr - u_ll))
end


@inline function flux_upwind(u_ll, u_rr, orientation, equation::HyperbolicDiffusionEquations2D)
  # Obtain left and right fluxes
  phi_ll, p_ll, q_ll = u_ll
  phi_rr, p_rr, q_rr = u_rr
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  # this is an optimized version of the application of the upwind dissipation matrix:
  #   dissipation = 0.5*R_n*|Λ|*inv(R_n)[[u]]
  λ_max = sqrt(equation.nu/equation.Tr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (phi_rr - phi_ll)
  if orientation == 1 # x-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (p_rr - p_ll)
    f3 = 1/2 * (f_ll[3] + f_rr[3])
  else # y-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2])
    f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (q_rr - q_ll)
  end

  return SVector(f1, f2, f3)
end


# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(u, element_id, invjacobian, cfl,
                     equation::HyperbolicDiffusionEquations2D, dg)
  λ_max = sqrt(equation.nu / equation.Tr)
  dt = cfl * 2 / (nnodes(dg) * invjacobian * λ_max)

  return dt
end


# Convert conservative variables to primitive
@inline cons2prim(u, equation::HyperbolicDiffusionEquations2D) = u

# Convert conservative variables to entropy found in I Do Like CFD, Too, Vol. 1
@inline function cons2entropy(u, equation::HyperbolicDiffusionEquations2D)
  phi, q1, q2 = u
  w1 = phi
  w2 = equation.Lr^2 * q1
  w3 = equation.Lr^2 * q2

  return SVector(w1, w2, w3)
end


# Calculate entropy for a conservative state `u` (here: same as total energy)
@inline entropy(u, equation::HyperbolicDiffusionEquations2D) = energy_total(u, equation)


# Calculate total energy for a conservative state `u`
@inline function energy_total(u, equation::HyperbolicDiffusionEquations2D)
  # energy function as found in equation (2.5.12) in the book "I Do Like CFD, Vol. 1"
  phi, q1, q2 = u
  return 0.5 * (phi^2 + equation.Lr^2 * (q1^2 + q2^2))
end
