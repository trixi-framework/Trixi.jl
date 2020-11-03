
@doc raw"""
    HyperbolicDiffusionEquations3D

The linear hyperbolic diffusion equations in three space dimensions.
A description of this system can be found in Sec. 2.5 of the book "I Do Like CFD, Too: Vol 1".
The book is freely available at http://www.cfdbooks.com/ and further analysis can be found in
the paper by Nishikawa [DOI: 10.1016/j.jcp.2007.07.029](https://doi.org/10.1016/j.jcp.2007.07.029)
"""
struct HyperbolicDiffusionEquations3D{RealT<:Real} <: AbstractHyperbolicDiffusionEquations{3, 4}
  Lr::RealT     # reference length scale
  inv_Tr::RealT # inverse of the reference time scale
  nu::RealT     # diffusion constant
  resid_tol::RealT # TODO Taal refactor, make this a parameter of a specialized steady-state solver
end

function HyperbolicDiffusionEquations3D(resid_tol; nu=1.0, Lr=inv(2pi))
  Tr = Lr^2 / nu
  HyperbolicDiffusionEquations3D(promote(Lr, inv(Tr), nu, resid_tol)...)
end

# TODO Taal refactor, remove old constructors and replace them with default values
function HyperbolicDiffusionEquations3D()
  # diffusion coefficient
  nu = parameter("nu", 1.0)
  # relaxation length scale
  Lr = parameter("Lr", 1.0/(2.0*pi))
  # relaxation time
  Tr = Lr*Lr/nu
  # stopping tolerance for the pseudotime "steady-state"
  resid_tol = parameter("resid_tol", 1e-12)
  HyperbolicDiffusionEquations3D(Lr, inv(Tr), nu, resid_tol)
end


get_name(::HyperbolicDiffusionEquations3D) = "HyperbolicDiffusionEquations3D"
varnames_cons(::HyperbolicDiffusionEquations3D) = @SVector ["phi", "q1", "q2", "q3"]
varnames_prim(::HyperbolicDiffusionEquations3D) = @SVector ["phi", "q1", "q2", "q3"]
default_analysis_quantities(::HyperbolicDiffusionEquations3D) = (:l2_error, :linf_error, :residual)
default_analysis_errors(::HyperbolicDiffusionEquations3D)     = (:l2_error, :linf_error, :residual)

@inline function residual_steady_state(du, ::HyperbolicDiffusionEquations3D)
  abs(du[1])
end


# Set initial conditions at physical location `x` for pseudo-time `t`
function initial_condition_poisson_periodic(x, t, equations::HyperbolicDiffusionEquations3D)
  # elliptic equation: -νΔϕ = f
  # depending on initial constant state, c, for phi this converges to the solution ϕ + c
  if iszero(t)
    phi = 0.0
    q1  = 0.0
    q2  = 0.0
    q3  = 0.0
  else
    phi =          sin(2 * pi * x[1]) * sin(2 * pi * x[2]) * sin(2 * pi * x[3])
    q1  = 2 * pi * cos(2 * pi * x[1]) * sin(2 * pi * x[2]) * sin(2 * pi * x[3])
    q2  = 2 * pi * sin(2 * pi * x[1]) * cos(2 * pi * x[2]) * sin(2 * pi * x[3])
    q3  = 2 * pi * sin(2 * pi * x[1]) * sin(2 * pi * x[2]) * cos(2 * pi * x[3])
  end
  return @SVector [phi, q1, q2, q3]
end

function initial_condition_poisson_nonperiodic(x, t, equations::HyperbolicDiffusionEquations3D)
  # elliptic equation: -νΔϕ = f
  if t == 0.0
    phi = 1.0
    q1  = 1.0
    q2  = 1.0
    q3  = 1.0
  else
    phi =  2.0 *      cos(pi * x[1]) * sin(2.0 * pi * x[2]) * sin(2.0 * pi * x[3]) + 2.0 # ϕ
    q1  = -2.0 * pi * sin(pi * x[1]) * sin(2.0 * pi * x[2]) * sin(2.0 * pi * x[3])   # ϕ_x
    q2  =  4.0 * pi * cos(pi * x[1]) * cos(2.0 * pi * x[2]) * sin(2.0 * pi * x[3])   # ϕ_y
    q3  =  4.0 * pi * cos(pi * x[1]) * sin(2.0 * pi * x[2]) * cos(2.0 * pi * x[3])   # ϕ_z
  end
  return @SVector [phi, q1, q2, q3]
end

function initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::HyperbolicDiffusionEquations3D)

  # Determine phi_x, phi_y
  G = 1.0 # gravitational constant
  C_grav = -4 * G / (3 * pi) # "3" is the number of spatial dimensions  # 2D: -2.0*G/pi
  A = 0.1 # perturbation coefficient must match Euler setup
  rho1 = A * sin(pi * (x[1] + x[2] + x[3] - t))
  # intialize with ansatz of gravity potential
  phi = C_grav * rho1
  q1  = C_grav * A * pi * cos(pi*(x[1] + x[2] + x[3] - t)) # = gravity acceleration in x-direction
  q2  = q1                                                 # = gravity acceleration in y-direction
  q3  = q1                                                 # = gravity acceleration in z-direction

  return @SVector [phi, q1, q2, q3]
end


function initial_condition_sedov_self_gravity(x, t, equations::HyperbolicDiffusionEquations3D)
  # for now just use constant initial condition for sedov blast wave (can likely be improved)

  phi = 0.0
  q1  = 0.0
  q2  = 0.0
  q3  = 0.0
  return @SVector [phi, q1, q2, q3]
end


# Apply boundary conditions
function boundary_condition_sedov_self_gravity(u_inner, orientation, direction, x, t,
                                                surface_flux_function,
                                                equations::HyperbolicDiffusionEquations3D)
  u_boundary = initial_condition_sedov_self_gravity(x, t, equations)

  # Calculate boundary flux
  if direction in (2, 4, 6) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end

function boundary_condition_poisson_nonperiodic(u_inner, orientation, direction, x, t,
                                                 surface_flux_function,
                                                 equations::HyperbolicDiffusionEquations3D)
  # elliptic equation: -νΔϕ = f
  phi =  2.0 *      cos(pi * x[1]) * sin(2.0 * pi * x[2]) * sin(2.0 * pi * x[3]) + 2.0 # ϕ
  q1  = -2.0 * pi * sin(pi * x[1]) * sin(2.0 * pi * x[2]) * sin(2.0 * pi * x[3])   # ϕ_x
  q2  =  4.0 * pi * cos(pi * x[1]) * cos(2.0 * pi * x[2]) * sin(2.0 * pi * x[3])   # ϕ_y
  q3  =  4.0 * pi * cos(pi * x[1]) * sin(2.0 * pi * x[2]) * cos(2.0 * pi * x[3])   # ϕ_z
  u_boundary = @SVector [phi, q1, q2, q3]

  # Calculate boundary flux
  if direction in (2, 4, 6) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


# Apply source terms
# TODO: Taal remove methods with the signature below
function source_terms_poisson_periodic(ut, u, x, element_id, t, n_nodes, equations::HyperbolicDiffusionEquations3D)
  # elliptic equation: -νΔϕ = f
  # analytical solution: phi = sin(2πx)*sin(2πy)*sin(2πz) and f = -12νπ^2 sin(2πx)*sin(2πy)*sin(2πz)
  @unpack inv_Tr = equations
  C = -12.0*equations.nu*pi*pi

  for k in 1:n_nodes, j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, k, element_id]
    x2 = x[2, i, j, k, element_id]
    x3 = x[3, i, j, k, element_id]
    tmp1 = sin(2 * pi * x1)
    tmp2 = sin(2 * pi * x2)
    tmp3 = sin(2 * pi * x3)
    ut[1, i, j, k, element_id] -= C*tmp1*tmp2*tmp3
    ut[2, i, j, k, element_id] -= inv_Tr * u[2, i, j, k, element_id]
    ut[3, i, j, k, element_id] -= inv_Tr * u[3, i, j, k, element_id]
    ut[4, i, j, k, element_id] -= inv_Tr * u[4, i, j, k, element_id]
  end

  return nothing
end

@inline function source_terms_poisson_periodic(u, x, t, equations::HyperbolicDiffusionEquations3D)
  # elliptic equation: -νΔϕ = f
  # analytical solution: phi = sin(2πx)*sin(2πy) and f = -8νπ^2 sin(2πx)*sin(2πy)
  @unpack inv_Tr = equations
  C = -12 * equations.nu * pi^2

  x1, x2, x3 = x
  tmp1 = sinpi(2 * x1)
  tmp2 = sinpi(2 * x2)
  tmp3 = sinpi(2 * x3)
  du1 = -C*tmp1*tmp2*tmp3
  du2 = -inv_Tr * u[2]
  du3 = -inv_Tr * u[3]
  du4 = -inv_Tr * u[4]

  return SVector(du1, du2, du3, du4)
end

# TODO: Taal remove methods with the signature below
function source_terms_poisson_nonperiodic(ut, u, x, element_id, t, n_nodes, equations::HyperbolicDiffusionEquations3D)
  # elliptic equation: -νΔϕ = f
  # analytical solution: ϕ = 2 cos(πx)sin(2πy)sin(2πz) + 2 and f = 18 π^2 cos(πx)sin(2πy)sin(2πz)
  @unpack inv_Tr = equations

  for k in 1:n_nodes, j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, k, element_id]
    x2 = x[2, i, j, k, element_id]
    x3 = x[3, i, j, k, element_id]
    ut[1, i, j, k, element_id] += 18 * pi^2 * cos(pi*x1) * sin(2.0*pi*x2) * sin(2.0*pi*x3)
    ut[2, i, j, k, element_id] -= inv_Tr * u[2, i, j, k, element_id]
    ut[3, i, j, k, element_id] -= inv_Tr * u[3, i, j, k, element_id]
    ut[4, i, j, k, element_id] -= inv_Tr * u[4, i, j, k, element_id]
  end

  return nothing
end

@inline function source_terms_poisson_nonperiodic(u, x, t, equations::HyperbolicDiffusionEquations3D)
  # elliptic equation: -νΔϕ = f
  # analytical solution: ϕ = 2 cos(πx)sin(2πy)sin(2πz) + 2 and f = 18 π^2 cos(πx)sin(2πy)sin(2πz)
  @unpack inv_Tr = equations

  x1, x2, x3 = x
  du1 = 18 * pi^2 * cospi(x1) * sinpi(2 * x2) * sinpi(2 * x3)
  du2 = -inv_Tr * u[2]
  du3 = -inv_Tr * u[3]
  du4 = -inv_Tr * u[4]

  return SVector(du1, du2, du3, du4)
end

# TODO: Taal remove methods with the signature below
function source_terms_harmonic(ut, u, x, element_id, t, n_nodes, equations::HyperbolicDiffusionEquations3D)
  # harmonic solution ϕ = (sinh(πx)sin(πy) + sinh(πy)sin(πx))/sinh(π), so f = 0
  @unpack inv_Tr = equations

  for k in 1:n_nodes, j in 1:n_nodes, i in 1:n_nodes
    ut[2, i, j, k, element_id] -= inv_Tr * u[2, i, j, k, element_id]
    ut[3, i, j, k, element_id] -= inv_Tr * u[3, i, j, k, element_id]
    ut[4, i, j, k, element_id] -= inv_Tr * u[4, i, j, k, element_id]
  end

  return nothing
end

@inline function source_terms_harmonic(u, x, t, equations::HyperbolicDiffusionEquations3D)
  # harmonic solution ϕ = (sinh(πx)sin(πy) + sinh(πy)sin(πx))/sinh(π), so f = 0
  @unpack inv_Tr = equations

  du1 = zero(u[1])
  du2 = -inv_Tr * u[2]
  du3 = -inv_Tr * u[3]
  du4 = -inv_Tr * u[4]

  return SVector(du1, du2, du3, du4)
end

# The coupled EOC test does not require additional sources
function source_terms_eoc_test_coupled_euler_gravity(ut, u, x, element_id, t, n_nodes, equations::HyperbolicDiffusionEquations3D)
  return source_terms_harmonic(ut, u, x, element_id, t, n_nodes, equations)
end


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equations::HyperbolicDiffusionEquations3D)
  phi, q1, q2, q3 = u

  if orientation == 1
    f1 = -equations.nu*q1
    f2 = -phi * equations.inv_Tr
    f3 = zero(phi)
    f4 = zero(phi)
  elseif orientation == 2
    f1 = -equations.nu*q2
    f2 = zero(phi)
    f3 = -phi * equations.inv_Tr
    f4 = zero(phi)
  else
    f1 = -equations.nu*q3
    f2 = zero(phi)
    f3 = zero(phi)
    f4 = -phi * equations.inv_Tr
  end

  return SVector(f1, f2, f3, f4)
end


@inline function flux_lax_friedrichs(u_ll, u_rr, orientation, equations::HyperbolicDiffusionEquations3D)
  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equations)
  f_rr = calcflux(u_rr, orientation, equations)

  λ_max = sqrt(equations.nu * equations.inv_Tr)

  return 0.5 * (f_ll + f_rr - λ_max * (u_rr - u_ll))
end


@inline function flux_upwind(u_ll, u_rr, orientation, equations::HyperbolicDiffusionEquations3D)
  # Obtain left and right fluxes
  phi_ll, q1_ll, q2_ll, q3_ll = u_ll
  phi_rr, q1_rr, q2_rr, q3_rr = u_rr
  f_ll = calcflux(u_ll, orientation, equations)
  f_rr = calcflux(u_rr, orientation, equations)

  # this is an optimized version of the application of the upwind dissipation matrix:
  #   dissipation = 0.5*R_n*|Λ|*inv(R_n)[[u]]
  λ_max = sqrt(equations.nu * equations.inv_Tr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (phi_rr - phi_ll)
  if orientation == 1 # x-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (q1_rr - q1_ll)
    f3 = 1/2 * (f_ll[3] + f_rr[3])
    f4 = 1/2 * (f_ll[4] + f_rr[4])
  elseif orientation == 2 # y-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2])
    f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (q2_rr - q2_ll)
    f4 = 1/2 * (f_ll[4] + f_rr[4])
  else # y-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2])
    f3 = 1/2 * (f_ll[3] + f_rr[3])
    f4 = 1/2 * (f_ll[4] + f_rr[4]) - 1/2 * λ_max * (q3_rr - q3_ll)
  end

  return SVector(f1, f2, f3, f4)
end


# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(u, element_id, invjacobian, cfl,
                     equations::HyperbolicDiffusionEquations3D, dg)
  λ_max = sqrt(equations.nu * equations.inv_Tr)
  dt = cfl * 2 / (nnodes(dg) * invjacobian * λ_max)

  return dt
end

@inline have_constant_speed(::HyperbolicDiffusionEquations3D) = Val(true)

@inline function max_abs_speeds(eq::HyperbolicDiffusionEquations3D)
  λ = sqrt(eq.nu * eq.inv_Tr)
  # FIXME Taal restore after Taam sync
  # return λ, λ, λ
  return λ, 0.0, 0.0
end


# Convert conservative variables to primitive
@inline cons2prim(u, equations::HyperbolicDiffusionEquations3D) = u


# Convert conservative variables to entropy found in I Do Like CFD, Too, Vol. 1
@inline function cons2entropy(u, equations::HyperbolicDiffusionEquations3D)
  phi, q1, q2, q3 = u
  w1 = phi
  w2 = equations.Lr^2 * q1
  w3 = equations.Lr^2 * q2
  w4 = equations.Lr^2 * q3

  return SVector(w1, w2, w3, w4)
end


# Calculate entropy for a conservative state `u` (here: same as total energy)
@inline entropy(u, equations::HyperbolicDiffusionEquations3D) = energy_total(u, equations)


# Calculate total energy for a conservative state `u`
@inline function energy_total(u, equations::HyperbolicDiffusionEquations3D)
  # energy function as found in equation (2.5.12) in the book "I Do Like CFD, Vol. 1"
  phi, q1, q2, q3 = u
  return 0.5 * (phi^2 + equations.Lr^2 * (q1^2 + q2^2 + q3^2))
end
