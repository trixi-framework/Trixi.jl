
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation
epsilon_relaxation = 1.0e-6
a1 = 10.0

equations_base = InviscidBurgersEquation1D()
velocities = (SVector(a1),)
equations = JinXinEquations(equations_base,epsilon_relaxation, velocities)
function initial_condition_linear_stability(x, t, equation::InviscidBurgersEquation1D)
    k = 1
    u = 2 + sinpi(k * (x[1] - 0.7))
    return prim2cons(SVector(u),equations)
end

basis = LobattoLegendreBasis(3)
solver = DGSEM(basis,Trixi.flux_upwind)


coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000)
initial_condition = Trixi.InitialConditionJinXin(initial_condition_linear_stability)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = Trixi.solve(ode, Trixi.SimpleIMEX(),
            dt = 1.0e-3, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks,maxiters=1e7);
summary_callback() # print the timer summary
