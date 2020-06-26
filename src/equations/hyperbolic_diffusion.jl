
@doc raw"""
    HyperbolicDiffusionEquations

The linear hyperbolic diffusion equations in two space dimensions.
A description of this system can be found in Sec. 2.5 of the book "I Do Like CFD, Too: Vol 1".
The book is freely available at http://www.cfdbooks.com/ and further analysis can be found in
the paper by Nishikawa [DOI: 10.1016/j.jcp.2007.07.029](https://doi.org/10.1016/j.jcp.2007.07.029)
"""
struct HyperbolicDiffusionEquations <: AbstractEquation{3}
  Lr::Float64
  Tr::Float64
  nu::Float64
  resid_tol::Float64
end

function HyperbolicDiffusionEquations()
  # diffusion coefficient
  nu = parameter("nu", 1.0)
  # relaxation length scale
  Lr = parameter("Lr", 1.0/(2.0*pi))
  # relaxation time
  Tr = Lr*Lr/nu
  # stopping tolerance for the pseudotime "steady-state"
  resid_tol = parameter("resid_tol", 1e-12)
  HyperbolicDiffusionEquations(Lr, Tr, nu, resid_tol)
end


get_name(::HyperbolicDiffusionEquations) = "HyperbolicDiffusionEquations"
varnames_cons(::HyperbolicDiffusionEquations) = @SVector ["phi", "p", "q"]
varnames_prim(::HyperbolicDiffusionEquations) = @SVector ["phi", "p", "q"]
default_analysis_quantities(::HyperbolicDiffusionEquations) = (:l2_error, :linf_error, :residual)


# Set initial conditions at physical location `x` for pseudo-time `t`
function initial_conditions_poisson_periodic(x, t, equation::HyperbolicDiffusionEquations)
  # elliptic equation: -νΔϕ = f
  # depending on initial constant state, c, for phi this converges to the solution ϕ + c
  if iszero(t)
    phi = 0.0
    p   = 0.0
    q   = 0.0
  else
    phi = sin(2.0*pi*x[1])*sin(2.0*pi*x[2])
    p   = 2*pi*cos(2.0*pi*x[1])*sin(2.0*pi*x[2])
    q   = 2*pi*sin(2.0*pi*x[1])*cos(2.0*pi*x[2])
  end
  return @SVector [phi, p, q]
end

function initial_conditions_poisson_nonperiodic(x, t, equation::HyperbolicDiffusionEquations)
  # elliptic equation: -νΔϕ = f
  if t == 0.0
    phi = 1.0
    p   = 1.0
    q   = 1.0
  else
    phi = 2.0*cos(pi*x[1])*sin(2.0*pi*x[2]) + 2.0 # ϕ
    p   = -2.0*pi*sin(pi*x[1])*sin(2.0*pi*x[2])   # ϕ_x
    q   = 4.0*pi*cos(pi*x[1])*cos(2.0*pi*x[2])    # ϕ_y
  end
  return @SVector [phi, p, q]
end

function initial_conditions_harmonic_nonperiodic(x, t, equation::HyperbolicDiffusionEquations)
  # elliptic equation: -νΔϕ = f
  if t == 0.0
    phi = 1.0
    p   = 1.0
    q   = 1.0
  else
    C   = 1.0/sinh(pi)
    phi = C*(sinh(pi*x[1])*sin(pi*x[2]) + sinh(pi*x[2])*sin(pi*x[1]))
    p   = C*pi*(cosh(pi*x[1])*sin(pi*x[2]) + sinh(pi*x[2])*cos(pi*x[1]))
    q   = C*pi*(sinh(pi*x[1])*cos(pi*x[2]) + cosh(pi*x[2])*sin(pi*x[1]))
  end
  return @SVector [phi, p, q]
end

function initial_conditions_jeans_instability(x, t, equation::HyperbolicDiffusionEquations)
  # gravity equation: -Δϕ = -4πGρ
  # Constants taken from the FLASH manual
  # https://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel.pdf
  rho0 = 1.5e7
  delta0 = 1e-3
  #
  phi = rho0*delta0 # constant background pertubation magnitude
  p   = 0.0
  q   = 0.0
  return @SVector [phi, p, q]
end

function initial_conditions_polytrope(x, t, equation::HyperbolicDiffusionEquations)
  inicenter = [0.0, 0.0] # must be same as in source terms
  r_soft = 0.001 # must be the same as in source terms

  # Calculate radius and radius with Plummer's softening to avoid singularity at r == 0.0
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)
  # r_plummer = (r^2 + r_soft^2) / r
  r_plummer = max(r, r_soft)

  # Determine phi_x, phi_y
  C = -2.0
  alpha = sqrt(2.0*pi)
  tmp_cos = alpha * r_plummer * cos(alpha * r_plummer)
  tmp_sin = sin(alpha * r_plummer)
  tmp = (tmp_cos - tmp_sin) / ( alpha * r_plummer^3)
  phi_x = C * x[1] * tmp # = gravity acceleration in x-direction
  phi_y = C * x[2] * tmp # = gravity acceleration in y-direction

  phi = C * sin(sqrt(2*pi) * r_plummer) / (sqrt(2*pi) * r_plummer)
  p   = phi_x
  q   = phi_y
  return @SVector [phi, p, q]
end

function initial_conditions_coupling_convergence_test(x, t, equation::HyperbolicDiffusionEquations)

  # Determine phi_x, phi_y
  G = 1.0 # gravitational constant
  C = -2.0*G/pi
  alpha = pi
  A = 0.1 # perturbation coefficient must match Euler setup
  rho1 = A*sin(alpha*(x[1] + x[2] - t))
  # intialize with ansatz of gravity potential
  phi = C * rho1
  p   = C * alpha * cos(alpha*(x[1] + x[2] - t)) # = gravity acceleration in x-direction
  q   = p                                        # = gravity acceleration in y-direction

  return @SVector [phi, p, q]
end

function initial_conditions_coupling_pulse_convergence_test(x, t, equation::HyperbolicDiffusionEquations)

  A = 1e-6
  w = 12.5

  eta   =      w * sin( pi * (x[1] - t) ) * sin( pi * (x[2] - t) )
  eta_x = pi * w * cos( pi * (x[1] - t) ) * sin( pi * (x[2] - t) )
  eta_y = pi * w * sin( pi * (x[1] - t) ) * cos( pi * (x[2] - t) )

  f   = A * eta   * exp(eta)
  f_x = A * eta_x * exp(eta) * (1.0 + eta)
  f_y = A * eta_y * exp(eta) * (1.0 + eta)

  G = 1.0 # gravitational constant
  C = -2.0*G/pi

  phi = C * f
  p   = C * f_x
  q   = C * f_y
  return @SVector [phi, p, q]
end

# Apply source terms
function source_terms_polytrope(ut, u, x, element_id, t, n_nodes, equation::HyperbolicDiffusionEquations)
  r_soft = 0.001 # must be the same as in initial conditions
  inicenter = [0.0, 0.0] # must be same as in initial conditions
  inv_Tr = inv(equation.Tr)

  for j in 1:n_nodes
    for i in 1:n_nodes
      # Calculate radius and radius with Plummer's softening to avoid singularity at r == 0.0
      x1 = x[1, i, j, element_id]
      x2 = x[2, i, j, element_id]
      x_norm = x1 - inicenter[1]
      y_norm = x2 - inicenter[2]
      r = sqrt(x_norm^2 + y_norm^2)
      # r_plummer = (r^2 + r_soft^2) / r
      r_plummer = max(r, r_soft)

      C = -2.0
      alpha = sqrt(2.0*pi)
      numerator = C*( (alpha^2*r_plummer^2 - 1.0)*sin(alpha*r_plummer) +
                       alpha*r_plummer*cos(alpha*r_plummer) )
      denominator = alpha * r_plummer^3

      ut[1, i, j, element_id] += numerator / denominator
      ut[2, i, j, element_id] -= inv_Tr * u[2, i, j, element_id]
      ut[3, i, j, element_id] -= inv_Tr * u[3, i, j, element_id]
    end
  end

  return nothing
end

function source_terms_poisson_periodic(ut, u, x, element_id, t, n_nodes, equation::HyperbolicDiffusionEquations)
  # elliptic equation: -νΔϕ = f
  # analytical solution: phi = sin(2πx)*sin(2πy) and f = -8νπ^2 sin(2πx)*sin(2πy)
  inv_Tr = inv(equation.Tr)
  C = -8.0*equation.nu*pi*pi

  for j in 1:n_nodes
    for i in 1:n_nodes
      x1 = x[1, i, j, element_id]
      x2 = x[2, i, j, element_id]
      tmp1 = sin(2.0*pi*x1)
      tmp2 = sin(2.0*pi*x2)
      ut[1, i, j, element_id] -= C*tmp1*tmp2
      ut[2, i, j, element_id] -= inv_Tr * u[2, i, j, element_id]
      ut[3, i, j, element_id] -= inv_Tr * u[3, i, j, element_id]
    end
  end

  return nothing
end

function source_terms_poisson_nonperiodic(ut, u, x, element_id, t, n_nodes, equation::HyperbolicDiffusionEquations)
  # elliptic equation: -νΔϕ = f
  # analytical solution: ϕ = 2cos(πx)sin(2πy) + 2 and f = 10π^2cos(πx)sin(2πy)
  inv_Tr = inv(equation.Tr)

  for j in 1:n_nodes
    for i in 1:n_nodes
      x1 = x[1, i, j, element_id]
      x2 = x[2, i, j, element_id]
      ut[1, i, j, element_id] += 10 * pi^2 * cos(pi*x1) * sin(2.0*pi*x2)
      ut[2, i, j, element_id] -= inv_Tr * u[2, i, j, element_id]
      ut[3, i, j, element_id] -= inv_Tr * u[3, i, j, element_id]
    end
  end

  return nothing
end

function source_terms_harmonic(ut, u, x, element_id, t, n_nodes, equation::HyperbolicDiffusionEquations)
  # harmonic solution ϕ = (sinh(πx)sin(πy) + sinh(πy)sin(πx))/sinh(π), so f = 0
  inv_Tr = inv(equation.Tr)

  for j in 1:n_nodes
    for i in 1:n_nodes
      ut[2, i, j, element_id] -= inv_Tr * u[2, i, j, element_id]
      ut[3, i, j, element_id] -= inv_Tr * u[3, i, j, element_id]
    end
  end

  return nothing
end

function source_terms_coupling_convergence_test(ut, u, x, element_id, t, n_nodes, equation::HyperbolicDiffusionEquations)
  # harmonic solution ϕ = (sinh(πx)sin(πy) + sinh(πy)sin(πx))/sinh(π), so f = 0
  inv_Tr = inv(equation.Tr)

  for j in 1:n_nodes
    for i in 1:n_nodes
      ut[2, i, j, element_id] -= inv_Tr * u[2, i, j, element_id]
      ut[3, i, j, element_id] -= inv_Tr * u[3, i, j, element_id]
    end
  end

  return nothing
end


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equation::HyperbolicDiffusionEquations)
  phi, p, q = u

  if orientation == 1
    f1 = -equation.nu*p
    f2 = -phi/equation.Tr
    f3 = zero(phi)
  else
    f1 = -equation.nu*q
    f2 = zero(phi)
    f3 = -phi/equation.Tr
  end

  return SVector(f1, f2, f3)
end


@inline function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::HyperbolicDiffusionEquations)
  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  λ_max = sqrt(equation.nu / equation.Tr)

  return 0.5 * (f_ll + f_rr - λ_max * (u_rr - u_ll))
end


@inline function flux_upwind(u_ll, u_rr, orientation, equation::HyperbolicDiffusionEquations)
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
function calc_max_dt(u, element_id, n_nodes, invjacobian, cfl,
                     equation::HyperbolicDiffusionEquations)
  dt = cfl * 2 / (invjacobian * sqrt(equation.nu/equation.Tr)) / n_nodes

  return dt
end


# Convert conservative variables to primitive
cons2prim(cons, equation::HyperbolicDiffusionEquations) =  cons

# Convert conservative variables to entropy found in I Do Like CFD, Too, Vol. 1
function cons2entropy(cons, n_nodes, n_elements, equation::HyperbolicDiffusionEquations)
  entropy = similar(cons)
  @. entropy[1, :, :, :] = cons[1, :, :, :]
  @. entropy[2, :, :, :] = equation.Lr*equation.Lr*cons[2, :, :, :]
  @. entropy[3, :, :, :] = equation.Lr*equation.Lr*cons[3, :, :, :]

  return entropy
end


# Calculate entropy for a conservative state `cons` (here: same as total energy)
@inline entropy(cons, equation::HyperbolicDiffusionEquations) = energy_total(cons, equation)


# Calculate total energy for a conservative state `cons`
@inline function energy_total(cons, equation::HyperbolicDiffusionEquations)
  # energy function as found in equation (2.5.12) in the book "I Do Like CFD, Vol. 1"
  return 0.5*(cons[1]^2 + equation.Lr^2 * (cons[2]^2 + cons[3]^2))
end
