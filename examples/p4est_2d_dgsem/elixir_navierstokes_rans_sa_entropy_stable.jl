using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# Entropy-stable discretization of the compressible RANS equations with the
# Spalart-Allmaras model in a closed box with no-slip adiabatic walls, started from
# perturbed initial data, similar to the robustness test of
# - D. Lodares, J. Manzanero, E. Ferrer, E. Valero (2022)
#   An entropy-stable discontinuous Galerkin approximation of the Spalart-Allmaras
#   turbulence model for the compressible Reynolds averaged Navier-Stokes equations
#   [DOI: 10.1016/j.jcp.2022.110998](https://doi.org/10.1016/j.jcp.2022.110998)
#
# The entropy-stable formulation combines
# - the entropy of `PassiveTracerEquations`, i.e., the Euler entropy plus ρν̃²,
# - an entropy-conservative volume flux (`FluxTracerEquationsCentral`) and an
#   entropy-stable surface flux (`FluxTracerEquationsUpwind`) for the turbulence variable,
# - gradients of the entropy variables for the parabolic terms,
# - the SA model without the density gradient term, which has no sign in the entropy balance,
# - entropy-stable wall boundary conditions (slip wall for the hyperbolic part,
#   entropy-variable no-slip adiabatic wall for the parabolic part).
# Without the turbulence production term, the total entropy cannot increase.

prandtl_number() = 0.72
mu() = 1.0e-3

equations = PassiveTracerEquations(CompressibleEulerEquations2D(1.4), n_tracers = 1)
model = SpalartAllmarasNeg(density_gradient_term = false)
equations_parabolic = CompressibleRANSDiffusion2D(equations, mu = mu(),
                                                  Prandtl = prandtl_number(),
                                                  model = model,
                                                  gradient_variables = GradientVariablesEntropy())

# Smooth perturbations of a state at rest
function initial_condition_perturbed(x, t, equations)
    rho = 1 + 0.2 * sinpi(3 * x[1] + 0.3) * cospi(2 * x[2] - 0.1)
    v1 = 0.3 * sinpi(2 * x[1]) * sinpi(5 * x[2] + 0.2)
    v2 = -0.3 * cospi(4 * x[1] - 0.4) * sinpi(3 * x[2])
    p = 1 + 0.3 * cospi(5 * x[1] + 0.1) * cospi(x[2] - 0.2)
    # ν̃ / ν changes sign to exercise the negative SA model
    nu_tilde = mu() / rho * (2 + 5 * sinpi(2 * x[1] - 0.2) * sinpi(4 * x[2] + 0.1))
    return prim2cons(SVector(rho, v1, v2, p, nu_tilde), equations)
end
initial_condition = initial_condition_perturbed

# Distance to the closest wall of the box [-1, 1]^2
wall_distance(x) = 1 - max(abs(x[1]), abs(x[2]))

surface_flux = FluxTracerEquationsUpwind(flux_lax_friedrichs)
volume_flux = FluxTracerEquationsCentral(flux_ranocha)
solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

trees_per_dimension = (8, 8)
mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                 periodicity = false)

boundary_conditions = (; x_neg = boundary_condition_slip_wall,
                       x_pos = boundary_condition_slip_wall,
                       y_neg = boundary_condition_slip_wall,
                       y_pos = boundary_condition_slip_wall)

velocity_bc = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_wall = BoundaryConditionNavierStokesWall(velocity_bc, heat_bc)
boundary_conditions_parabolic = (; x_neg = boundary_condition_wall,
                                 x_pos = boundary_condition_wall,
                                 y_neg = boundary_condition_wall,
                                 y_pos = boundary_condition_wall)

source_terms_parabolic = SourceTermsSpalartAllmaras(wall_distance)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic),
                                             source_terms_parabolic = source_terms_parabolic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))
alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(); abstol = 1.0e-6, reltol = 1.0e-6,
            ode_default_options()..., callback = callbacks)
