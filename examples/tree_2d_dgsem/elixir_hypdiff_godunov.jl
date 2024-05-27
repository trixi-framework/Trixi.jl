
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the hyperbolic diffusion equations

equations = HyperbolicDiffusionEquations2D()

function initial_condition_poisson_periodic(x, t, equations::HyperbolicDiffusionEquations2D)
    # elliptic equation: -νΔϕ = f
    # depending on initial constant state, c, for phi this converges to the solution ϕ + c
    if iszero(t)
        phi = 0.0
        q1 = 0.0
        q2 = 0.0
    else
        phi = sin(2.0 * pi * x[1]) * sin(2.0 * pi * x[2])
        q1 = 2 * pi * cos(2.0 * pi * x[1]) * sin(2.0 * pi * x[2])
        q2 = 2 * pi * sin(2.0 * pi * x[1]) * cos(2.0 * pi * x[2])
    end
    return SVector(phi, q1, q2)
end
initial_condition = initial_condition_poisson_periodic

@inline function source_terms_poisson_periodic(u, x, t,
                                               equations::HyperbolicDiffusionEquations2D)
    # elliptic equation: -νΔϕ = f
    # analytical solution: phi = sin(2πx)*sin(2πy) and f = -8νπ^2 sin(2πx)*sin(2πy)
    @unpack inv_Tr = equations
    C = -8 * equations.nu * pi^2

    x1, x2 = x
    tmp1 = sinpi(2 * x1)
    tmp2 = sinpi(2 * x2)
    du1 = -C * tmp1 * tmp2
    du2 = -inv_Tr * u[2]
    du3 = -inv_Tr * u[3]

    return SVector(du1, du2, du3)
end

volume_flux = flux_central
solver = DGSEM(polydeg = 4, surface_flux = flux_godunov,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_poisson_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
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
