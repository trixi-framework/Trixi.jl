
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advectionvelocity)

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

initial_condition = initial_condition_gauss
boundary_condition = BoundaryConditionDirichlet(initial_condition)

boundary_conditions = Dict(
  :all => boundary_condition
)

# Mapping as described in https://arxiv.org/abs/2012.12040, but with less warping.
# The original mapping applied to this unstructured mesh creates extreme angles,
# which require a high resolution for proper results.
function mapping(xi, eta, zeta)
  # Don't transform input variables between -1 and 1 onto [0,3] to obtain curved boundaries
  # xi = 1.5 * xi_ + 1.5
  # eta = 1.5 * eta_ + 1.5
  # zeta = 1.5 * zeta_ + 1.5

  y = eta + 1/4 * (cos(1.5 * pi * (2 * xi - 3)/3) *
                   cos(0.5 * pi * (2 * eta - 3)/3) *
                   cos(0.5 * pi * (2 * zeta - 3)/3))

  x = xi + 1/4 * (cos(0.5 * pi * (2 * xi - 3)/3) *
                  cos(2 * pi * (2 * y - 3)/3) *
                  cos(0.5 * pi * (2 * zeta - 3)/3))

  z = zeta + 1/4 * (cos(0.5 * pi * (2 * x - 3)/3) *
                    cos(pi * (2 * y - 3)/3) *
                    cos(0.5 * pi * (2 * zeta - 3)/3))

  # Transform the weird deformed cube to be approximately the size of [-5,5]^3 to match IC
  return SVector(5 * x, 5 * y, 5 * z)
end

# Unstructured mesh with 48 cells of the cube domain [-1, 1]^3
mesh_file = joinpath(@__DIR__, "cube_unstructured_2.inp")
isfile(mesh_file) || download("https://gist.githubusercontent.com/efaulhaber/b8df0033798e4926dec515fc045e8c2c/raw/b9254cde1d1fb64b6acc8416bc5ccdd77a240227/cube_unstructured_2.inp",
                              mesh_file)

mesh = P4estMesh{3}(mesh_file, polydeg=3,
                    mapping=mapping,
                    initial_refinement_level=1)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 8.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first),
                                      base_level=1,
                                      med_level=2, med_threshold=0.1,
                                      max_level=3, max_threshold=0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=1.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_restart,
                        save_solution,
                        amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
