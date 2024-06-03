
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations


epsilon_relaxation = 1.0e-6

equations_base = CompressibleEulerEquations1D(1.4)
velocities = (SVector(a1, a2, a3),)
equations = JinXinEquations(equations_base, epsilon_relaxation, velocities)

function initial_condition_density_wave(x, t, equations::CompressibleEulerEquations1D)
    v1 = 0.1
    rho = 1 + 0.98 * sinpi(2 * (x[1] - t * v1))
    p = 20
    return prim2cons(SVector(rho, v1, p),equations)
end


initial_condition = Trixi.InitialConditionJinXin(initial_condition_density_wave)
polydeg = 3
#basis = LobattoLegendreBasis(polydeg; polydeg_projection = 0)
basis = LobattoLegendreBasis(polydeg; polydeg_projection = 6)

volume_integral = VolumeIntegralWeakForm()
#solver = DGSEM(basis, Trixi.flux_upwind,VolumeIntegralWeakForm())
solver = DGSEM(basis, Trixi.flux_upwind)

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)


alive_callback = AliveCallback(analysis_interval = analysis_interval)
save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = Trixi.solve(ode, Trixi.SimpleIMEX(),
            dt = 1e-3, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary