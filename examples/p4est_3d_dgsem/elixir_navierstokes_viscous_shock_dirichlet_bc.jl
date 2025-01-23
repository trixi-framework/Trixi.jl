using OrdinaryDiffEq
using Trixi

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
# We want kappa = cp * mu = mu_bar to ensure constant enthalpy
prandtl_number() = 3 / 4

### Free choices: ###
const gamma = 5 / 3

# In Margolin et al., the Navier-Stokes equations are given for an 
# isotropic stress tensor τ, i.e., ∇ ⋅ τ = μ Δu 
const mu_isotropic = 0.15
const mu_bar = mu_isotropic / (gamma - 1) # Re-scaled viscosity

const rho_0 = 1
const v = 1 # Shock speed

const domain_length = 4.0

### Derived quantities ###

const Ma = 2 / sqrt(3 - gamma) # Mach number for alpha = 0.5 (see Margolin et al.)
const c_0 = v / Ma # Speed of sound ahead of the shock

# From constant enthalpy condition
const p_0 = c_0^2 * rho_0 / gamma

const l = mu_bar / (rho_0 * v) * 2 * gamma / (gamma + 1) # Appropriate length scale

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
    y = x[1] - v * t # Translated coordinate

    # Coordinate transformation. See eq. (33) in Margolin et al. (2017)
    chi = 2 * exp(y / (2 * l))

    w = 1 + 1 / (2 * chi^2) * (1 - sqrt(1 + 2 * chi^2))

    rho = rho_0 / w
    u = v * (1 - w)
    p = p_0 * 1 / w * (1 + (gamma - 1) / 2 * Ma^2 * (1 - w^2))

    return prim2cons(SVector(rho, u, 0, 0, p), equations)
end
initial_condition = initial_condition_viscous_shock

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

equations = CompressibleEulerEquations3D(gamma)

# Trixi implements the stress tensor in deviatoric form, thus we need to 
# convert the "isotropic viscosity" to the "deviatoric viscosity"
const mu_deviatoric = mu_bar * 3 / 4
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu_deviatoric,
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

solver = DGSEM(polydeg = 3, surface_flux = flux_hlle)

coordinates_min = (-domain_length / 2, -domain_length / 2, -domain_length / 2)
coordinates_max = (domain_length / 2, domain_length / 2, domain_length / 2)

trees_per_dimension = (8, 2, 2)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 3, initial_refinement_level = 0,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = (false, true, true))

### Inviscid boundary conditions ###

# Prescribe pure influx based on initial conditions
function boundary_condition_inflow(u_inner, normal_direction::AbstractVector, x, t,
                                   surface_flux_function,
                                   equations::CompressibleEulerEquations3D)
    u_cons = initial_condition_viscous_shock(x, t, equations)
    flux = Trixi.flux(u_cons, normal_direction, equations)

    return flux
end

# Completely free outflow
function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                    surface_flux_function,
                                    equations::CompressibleEulerEquations3D)
    # Calculate the boundary flux entirely from the internal solution state
    flux = Trixi.flux(u_inner, normal_direction, equations)

    return flux
end

boundary_conditions = Dict(:x_neg => boundary_condition_inflow,
                           :x_pos => boundary_condition_outflow)

### Viscous boundary conditions ###
# For the viscous BCs, we use the known analytical solution 
boundary_condition_parabolic_dirichlet = BoundaryConditionDirichlet(initial_condition)

boundary_conditions_parabolic = Dict(:x_neg => boundary_condition_parabolic_dirichlet,
                                     :x_pos => boundary_condition_parabolic_dirichlet)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

alive_callback = AliveCallback(alive_interval = 10)

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)

summary_callback() # print the timer summary
