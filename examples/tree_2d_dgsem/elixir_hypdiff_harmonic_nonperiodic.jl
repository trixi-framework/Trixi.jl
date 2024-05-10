
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the hyperbolic diffusion equations

equations = HyperbolicDiffusionEquations2D()

@inline function initial_condition_harmonic_nonperiodic(x, t,
                                                        equations::HyperbolicDiffusionEquations2D)
    # elliptic equation: -ν Δϕ = 0 in Ω, u = g on ∂Ω
    if t == 0.0
        phi = 1.0
        q1 = 1.0
        q2 = 1.0
    else
        C = inv(sinh(pi))
        sinpi_x1, cospi_x1 = sincos(pi * x[1])
        sinpi_x2, cospi_x2 = sincos(pi * x[2])
        sinh_pix1 = sinh(pi * x[1])
        cosh_pix1 = cosh(pi * x[1])
        sinh_pix2 = sinh(pi * x[2])
        cosh_pix2 = cosh(pi * x[2])
        phi = C * (sinh_pix1 * sinpi_x2 + sinh_pix2 * sinpi_x1)
        q1 = C * pi * (cosh_pix1 * sinpi_x2 + sinh_pix2 * cospi_x1)
        q2 = C * pi * (sinh_pix1 * cospi_x2 + cosh_pix2 * sinpi_x1)
    end
    return SVector(phi, q1, q2)
end
initial_condition = initial_condition_harmonic_nonperiodic

boundary_conditions = BoundaryConditionDirichlet(initial_condition)

solver = DGSEM(polydeg = 4, surface_flux = flux_godunov)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions,
                                    source_terms = source_terms_harmonic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

resid_tol = 5.0e-12
steady_state_callback = SteadyStateCallback(abstol = resid_tol, reltol = 0.0)

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback, steady_state_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
