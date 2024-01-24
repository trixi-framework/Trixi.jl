
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_gauss

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:all => boundary_condition)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-5.0, 5 * s - 5.0)
f2(s) = SVector(5.0, 5 * s + 5.0)
f3(s) = SVector(5 * s, -5.0 + 5 * sin(0.5 * pi * s))
f4(s) = SVector(5 * s, 5.0 + 5 * sin(0.5 * pi * s))
faces = (f1, f2, f3, f4)

# This creates a mapping that transforms [-1, 1]^2 to the domain with the faces defined above.
# It generally doesn't work for meshes loaded from mesh files because these can be meshes
# of arbitrary domains, but the mesh below is specifically built on the domain [-1, 1]^2.
Trixi.validate_faces(faces)
mapping_flag = Trixi.transfinite_mapping(faces)

# Unstructured mesh with 24 cells of the square domain [-1, 1]^n
mesh_file = Trixi.download("https://gist.githubusercontent.com/efaulhaber/63ff2ea224409e55ee8423b3a33e316a/raw/7db58af7446d1479753ae718930741c47a3b79b7/square_unstructured_2.inp",
                           joinpath(@__DIR__, "square_unstructured_2.inp"))

mesh = P4estMesh{2}(mesh_file, polydeg = 3,
                    mapping = mapping_flag,
                    initial_refinement_level = 1)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                      base_level = 1,
                                      med_level = 2, med_threshold = 0.1,
                                      max_level = 3, max_threshold = 0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 0.7)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart, save_solution,
                        amr_callback, stepsize_callback);

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
