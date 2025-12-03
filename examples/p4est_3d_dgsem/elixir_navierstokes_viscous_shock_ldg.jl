using OrdinaryDiffEqLowStorageRK
using Trixi

# This is the classic 1D viscous shock wave problem with analytical solution
# for a special value of the Prandtl number.
# For more info, see "elixir_navierstokes_viscous_shock.jl"

### Fixed parameters ###

alpha = 0.5 # Fixes the Mach number
prandtl_number() = 3 / 4 # We want kappa = cp * mu = mu_bar to ensure constant enthalpy

### Free choices: ###
gamma() = 5 / 3

# In Margolin et al., the Navier-Stokes equations are given for an
# isotropic stress tensor τ, i.e., ∇ ⋅ τ = μ Δu
mu_isotropic() = 0.15
mu_bar() = mu_isotropic() / (gamma() - 1) # Re-scaled viscosity

rho_0() = 1
v() = 1 # Shock speed

domain_length = 4.0

### Derived quantities ###

Ma() = 2 / sqrt(3 - gamma()) # Mach number for alpha = 0.5
c_0() = v() / Ma() # Speed of sound ahead of the shock

p_0() = c_0()^2 * rho_0() / gamma() # From constant enthalpy condition

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

    return prim2cons(SVector(rho, u, 0, 0, p), equations)
end
initial_condition = initial_condition_viscous_shock

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

equations = CompressibleEulerEquations3D(gamma())

# Trixi implements the stress tensor in deviatoric form, thus we need to
# convert the "isotropic viscosity" to the "deviatoric viscosity"
mu_deviatoric() = mu_bar() * 3 / 4
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu_deviatoric(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = flux_hlle)

# Affine type mapping to take the [-1,1]^3 domain
# and warp it as described in https://arxiv.org/abs/2012.12040
function mapping(xi, eta, zeta)
    y_ = eta + 1 / 6 * (cos(1.5 * pi * xi) * cos(0.5 * pi * eta) * cos(0.5 * pi * zeta))
    x_ = xi + 1 / 6 * (cos(0.5 * pi * xi) * cos(2 * pi * y_) * cos(0.5 * pi * zeta))
    z_ = zeta + 1 / 6 * (cos(0.5 * pi * x_) * cos(pi * y_) * cos(0.5 * pi * zeta))

    # Map from [-1, 1]^3 to [-domain_length/2, domain_length/2]^3
    x = domain_length / 2 * x_
    y = domain_length / 2 * y_
    z = domain_length / 2 * z_

    return SVector(x, y, z)
end

trees_per_dimension = (5, 3, 3)
mesh = P4estMesh(trees_per_dimension, polydeg = polydeg,
                 mapping = mapping, periodicity = (false, true, true))

### Inviscid boundary conditions ###

# Prescribe pure influx based on initial conditions
function boundary_condition_inflow(u_inner, normal_direction::AbstractVector, x, t,
                                   surface_flux_function,
                                   equations::CompressibleEulerEquations3D)
    u_cons = initial_condition_viscous_shock(x, t, equations)
    return flux(u_cons, normal_direction, equations)
end

# Completely free outflow
function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                    surface_flux_function,
                                    equations::CompressibleEulerEquations3D)
    # Calculate the boundary flux entirely from the internal solution state
    return flux(u_inner, normal_direction, equations)
end

boundary_conditions = Dict(:x_neg => boundary_condition_inflow,
                           :x_pos => boundary_condition_outflow)

### Viscous boundary conditions ###
# For the viscous BCs, we use the known analytical solution
velocity_bc = NoSlip() do x, t, equations_parabolic
    velocity(initial_condition_viscous_shock(x, t, equations_parabolic),
             equations_parabolic)
end

heat_bc = Isothermal() do x, t, equations_parabolic
    temperature(initial_condition_viscous_shock(x, t, equations_parabolic),
                equations_parabolic)
end

boundary_condition_parabolic = BoundaryConditionNavierStokesWall(velocity_bc, heat_bc)

boundary_conditions_parabolic = Dict(:x_neg => boundary_condition_parabolic,
                                     :x_pos => boundary_condition_parabolic)

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
            dt = 1e-3, ode_default_options()..., callback = callbacks)
