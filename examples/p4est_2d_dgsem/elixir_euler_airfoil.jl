
using Trixi
using OrdinaryDiffEq, Plots

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_constant

d = 3

solver = DGSEM(polydeg = d, surface_flux = flux_lax_friedrichs)

base_path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Airfoil_geo/"

mesh_file = base_path * "NACA.inp"

boundary_symbols = [:PhysicalLine10, :PhysicalLine20, :PhysicalLine30, :PhysicalLine40]

mesh = P4estMesh{2}(mesh_file, polydeg = d, boundary_symbols=boundary_symbols)
boundary_conditions = Dict(:PhysicalLine10 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalLine20 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalLine30 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalLine40 => BoundaryConditionDirichlet(initial_condition))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 2.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary

plot(sol)

pd = PlotData2D(sol)
plot(pd["rho"])
plot!(getmesh(pd))