
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_constant

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

# Mapping as described in https://arxiv.org/abs/2012.12040, but reduced to 2D
function mapping(xi_, eta_)
  # Transform input variables between -1 and 1 onto [0,3]
  xi = 1.5 * xi_ + 1.5
  eta = 1.5 * eta_ + 1.5

  y = eta + 3/8 * (cos(1.5 * pi * (2 * xi - 3)/3) *
                   cos(0.5 * pi * (2 * eta - 3)/3))

  x = xi + 3/8 * (cos(0.5 * pi * (2 * xi - 3)/3) *
                  cos(2 * pi * (2 * y - 3)/3))

  return SVector(x, y)
end

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

# Unstructured mesh with 48 cells of the square domain [-1, 1]^n
mesh_file = joinpath(@__DIR__, "square_unstructured_1.inp")
isfile(mesh_file) || download("https://gist.githubusercontent.com/efaulhaber/a075f8ec39a67fa9fad8f6f84342cbca/raw/a7206a02ed3a5d3cadacd8d9694ac154f9151db7/square_unstructured_1.inp",
                              mesh_file)

# Map the unstructured mesh with the mapping above
mesh = P4estMesh(mesh_file, polydeg=3, mapping=mapping, initial_refinement_level=1)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=Dict(
                                      :all => BoundaryConditionDirichlet(initial_condition)
                                    ))


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=2.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
