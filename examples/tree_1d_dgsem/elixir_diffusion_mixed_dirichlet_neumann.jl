using Trixi

###############################################################################
# semidiscretization of the pure diffusion equation with mixed Dirichlet-Neumann BCs

diffusivity() = 0.5
forcing_amplitude() = 0.4
forcing_frequency() = 4.0
dirichlet_mean() = 1.0
angular_frequency() = 2 * pi * forcing_frequency()

# The equations object still uses an auxiliary 1D hyperbolic scalar equation for variable metadata.
equations_hyperbolic = LinearScalarAdvectionEquation1D(0.0)
equations = LaplaceDiffusion1D(diffusivity(), equations_hyperbolic)

solver = DGSEM(polydeg = 3, surface_flux = flux_central)

mesh = TreeMesh((0.0,), (1.0,),
                initial_refinement_level = 3,
                periodicity = false,
                n_cells_max = 30_000)

complex_wavenumber() = sqrt(im * angular_frequency() / diffusivity())
harmonic_shape(x) = cosh(complex_wavenumber() * (1 - x)) / cosh(complex_wavenumber())

exact_solution(x, t) = dirichlet_mean() +
                       imag(forcing_amplitude() * exp(im * angular_frequency() * t) *
                            harmonic_shape(x[1]))

initial_condition(x, t, equations) = SVector(exact_solution(x, t))

dirichlet_left(x, t) = dirichlet_mean() + forcing_amplitude() * sin(angular_frequency() * t)
neumann_right(x, t) = 0.0

boundary_conditions = (; x_neg = BoundaryConditionDirichlet((x, t, equations) -> SVector(dirichlet_left(x, t))),
                       x_pos = BoundaryConditionNeumann((x, t, equations) -> SVector(neumann_right(x, t))))

semi = SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                   boundary_conditions = boundary_conditions,
                                   solver_parabolic = ViscousFormulationLocalDG())

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.25)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_integrals = ())

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = Trixi.solve(ode, Trixi.CarpenterKennedy2N54(); dt = 5.0e-5, adaptive = false,
                  ode_default_options()..., callback = callbacks)
