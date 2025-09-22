using Trixi
using OrdinaryDiffEqBDF # BDF subpackage exports IMEX methods
using LinearSolve # For Jacobian-free Newton-Krylov (GMRES) solver
using ADTypes # To access the types choosing how to evaluate Jacobian-vector products

# This is the classic 1D viscous shock wave problem with analytical solution
# for a special value of the Prandtl number.
# The original references are:
#
# - R. Becker (1922)
#   Stoßwelle und Detonation.
#   [DOI: 10.1007/BF01329605](https://doi.org/10.1007/BF01329605)
#
#   English translations:
#   Impact waves and detonation. Part I.
#   https://ntrs.nasa.gov/api/citations/19930090862/downloads/19930090862.pdf
#   Impact waves and detonation. Part II.
#   https://ntrs.nasa.gov/api/citations/19930090863/downloads/19930090863.pdf
#
# - M. Morduchow, P. A. Libby (1949)
#   On a Complete Solution of the One-Dimensional Flow Equations
#   of a Viscous, Head-Conducting, Compressible Gas
#   [DOI: 10.2514/8.11882](https://doi.org/10.2514/8.11882)
#
#
# The particular problem considered here is described in
# - L. G. Margolin, J. M. Reisner, P. M. Jordan (2017)
#   Entropy in self-similar shock profiles
#   [DOI: 10.1016/j.ijnonlinmec.2017.07.003](https://doi.org/10.1016/j.ijnonlinmec.2017.07.003)

### Fixed parameters ###

# Special value for which nonlinear solver can be omitted
# Corresponds essentially to fixing the Mach number
alpha = 0.5
# We want kappa = cp * mu = mu_bar to ensure constant enthalpy
prandtl_number() = 1

### Free choices: ###
gamma() = 5 / 3

mu() = 0.15
mu_bar() = mu() / (gamma() - 1) # Re-scaled viscosity

rho_0() = 1
v() = 1 # Shock speed

domain_length = 4.0

### Derived quantities ###

Ma() = 2 / sqrt(3 - gamma()) # Mach number for alpha = 0.5
c_0() = v() / Ma() # Speed of sound ahead of the shock

# From constant enthalpy condition
p_0() = c_0()^2 * rho_0() / gamma()

l() = mu_bar() / (rho_0() * v()) * 2 * gamma() / (gamma() + 1) # Appropriate length scale

"""
    initial_condition_viscous_shock(x, t, equations)

Classic 1D viscous shock wave problem with analytical solution
for a special value of the Prandtl number.
The version implemented here is described in
- L. G. Margolin, J. M. Reisner, P. M. Jordan (2017)
  Entropy in self-similar shock profiles
  [DOI: 10.1016/j.ijnonlinmec.2017.07.003](https://doi.org/10.1016/j.ijnonlinmec.2017.07.003)
"""
function initial_condition_viscous_shock(x, t, equations)
    y = x[1] - v() * t # Translated coordinate

    # Coordinate transformation. See eq. (33) in Margolin et al. (2017)
    chi = 2 * exp(y / (2 * l()))

    w = 1 + 1 / (2 * chi^2) * (1 - sqrt(1 + 2 * chi^2))

    rho = rho_0() / w
    u = v() * (1 - w)
    p = p_0() * 1 / w * (1 + (gamma() - 1) / 2 * Ma()^2 * (1 - w^2))

    return prim2cons(SVector(rho, u, p), equations)
end
initial_condition = initial_condition_viscous_shock

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

equations = CompressibleEulerEquations1D(gamma())
equations_parabolic = CompressibleNavierStokesDiffusion1D(equations, mu = mu_bar(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

solver = DGSEM(polydeg = 3, surface_flux = flux_hlle)

coordinates_min = -domain_length / 2
coordinates_max = domain_length / 2

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                periodicity = false,
                n_cells_max = 30_000)

### Inviscid boundary conditions ###

# Prescribe pure influx based on initial conditions
function boundary_condition_inflow(u_inner, orientation::Integer, normal_direction, x, t,
                                   surface_flux_function,
                                   equations::CompressibleEulerEquations1D)
    u_cons = initial_condition_viscous_shock(x, t, equations)
    return flux(u_cons, orientation, equations)
end

boundary_conditions = (; x_neg = boundary_condition_inflow,
                       x_pos = boundary_condition_do_nothing)

### Viscous boundary conditions ###
# For the viscous BCs, we use the known analytical solution
velocity_bc = NoSlip() do x, t, equations_parabolic
    Trixi.velocity(initial_condition_viscous_shock(x,
                                                   t,
                                                   equations_parabolic),
                   equations_parabolic)
end

heat_bc = Isothermal() do x, t, equations_parabolic
    Trixi.temperature(initial_condition_viscous_shock(x,
                                                      t,
                                                      equations_parabolic),
                      equations_parabolic)
end

boundary_condition_parabolic = BoundaryConditionNavierStokesWall(velocity_bc, heat_bc)

boundary_conditions_parabolic = (; x_neg = boundary_condition_parabolic,
                                 x_pos = boundary_condition_parabolic)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = ViscousFormulationLocalDG(),
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################

tspan = (0.0, 0.75)
# For hyperbolic-parabolic problems, this results in a SciML SplitODEProblem, see e.g.
# https://docs.sciml.ai/DiffEqDocs/stable/types/split_ode_types/#SciMLBase.SplitODEProblem
# These exactly fit IMEX (implicit-explicit) integrators
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

alive_callback = AliveCallback(alive_interval = 100)

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################

# Tolerances for GMRES residual, see https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#Krylov.gmres
atol_lin_solve = 1e-4
rtol_lin_solve = 1e-4

# Jacobian-free Newton-Krylov (GMRES) solver
linsolve = KrylovJL_GMRES(atol = atol_lin_solve, rtol = rtol_lin_solve)

# Choice of method:
# https://docs.sciml.ai/OrdinaryDiffEq/stable/imex/IMEXBDF/#Solver-Selection-Guide
# higher order methods (`SBDF3` and `SBDF4`) have trouble converging.
#
# Use IMEX Runge-Kutta method with Jacobian-free (!) Newton-Krylov (GMRES) implicit solver, see
# https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/#Using-Jacobian-Free-Newton-Krylov
ode_alg = SBDF2(autodiff = AutoFiniteDiff(), linsolve = linsolve)

sol = solve(ode, ode_alg; dt = 0.05, # Fixed timestep
            ode_default_options()..., callback = callbacks);
