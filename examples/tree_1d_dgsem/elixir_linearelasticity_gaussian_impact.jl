using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear elasticity equations

# Material parameters corresponding to steel
rho = 7800.0 # kg/m³
lambda = 9.3288e10
mu = lambda
equations = LinearElasticityEquations1D(rho = rho, mu = mu, lambda = lambda)

basis = LobattoLegendreBasis(5)

# Use subcell shock capturing techniques to weaken oscillations at discontinuities
alpha_max = 0.4 # This controls the amount of dissipation added (larger = more dissipation)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = alpha_max,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = first)

volume_flux = flux_central
surface_flux = flux_lax_friedrichs

volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinate_min = -1.0
coordinate_max = 1.0

# E.g. hammer impact width of 20 cm
impact_width() = 0.2
# Refine the impact region to be able to represent the initial condition without oscillations
refinement_patches = ((type = "box",
                       coordinates_min = (-impact_width() / 2,),
                       coordinates_max = (impact_width() / 2,)),)

# Make sure to turn periodicity explicitly off as special boundary conditions are specified
mesh = TreeMesh(coordinate_min, coordinate_max,
                initial_refinement_level = 5,
                n_cells_max = 10_000,
                refinement_patches = refinement_patches,
                periodicity = false)

function initial_condition_gaussian_impact(x, t, equations::LinearElasticityEquations1D)
    amplitude = 1e6 # 1 MPa stress
    edge_smoothing = 0.01 # Controls sharpness of edges (smaller = sharper)

    # Smooth rectangle using tanh transitions
    # Left edge: transition from 0 to 1 at x = -width/2
    left_transition = 0.5 * (1 + tanh((x[1] + impact_width() / 2) / edge_smoothing))
    # Right edge: transition from 1 to 0 at x = +width/2  
    right_transition = 0.5 * (1 - tanh((x[1] - impact_width() / 2) / edge_smoothing))

    # Combine both transitions to create smooth rectangle
    sigma = amplitude * left_transition * right_transition

    v = 0 # Initial displacement velocity is zero
    return SVector(v, sigma)
end

###############################################################################
# Specify non-periodic boundary conditions

boundary_condition_dirichlet = BoundaryConditionDirichlet(initial_condition_gaussian_impact)

boundary_conditions = (x_neg = boundary_condition_dirichlet,
                       x_pos = boundary_condition_dirichlet)

initial_condition = initial_condition_gaussian_impact

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1e-4) # Relatively short simulation time due to high wave speeds (~5990 m/s)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 42.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
