
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_constant

boundary_conditions = Dict(
  :all => BoundaryConditionDirichlet(initial_condition)
)

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs,
               volume_integral=VolumeIntegralWeakForm())

# Mapping as described in https://arxiv.org/abs/2012.12040 but with less warping.
function mapping(xi_, eta_, zeta_)
  # Transform input variables between -1 and 1 onto [0,3]
  xi = 1.5 * xi_ + 1.5
  eta = 1.5 * eta_ + 1.5
  zeta = 1.5 * zeta_ + 1.5

  y = eta + 1/6 * (cos(1.5 * pi * (2 * xi - 3)/3) *
                   cos(0.5 * pi * (2 * eta - 3)/3) *
                   cos(0.5 * pi * (2 * zeta - 3)/3))

  x = xi + 1/6 * (cos(0.5 * pi * (2 * xi - 3)/3) *
                  cos(2 * pi * (2 * y - 3)/3) *
                  cos(0.5 * pi * (2 * zeta - 3)/3))

  z = zeta + 1/6 * (cos(0.5 * pi * (2 * x - 3)/3) *
                    cos(pi * (2 * y - 3)/3) *
                    cos(0.5 * pi * (2 * zeta - 3)/3))

  return SVector(x, y, z)
end

# Unstructured mesh with 68 cells of the cube domain [-1, 1]^3
mesh_file = joinpath(@__DIR__, "cube_unstructured_1.inp")
isfile(mesh_file) || download("https://gist.githubusercontent.com/efaulhaber/d45c8ac1e248618885fa7cc31a50ab40/raw/37fba24890ab37cfa49c39eae98b44faf4502882/cube_unstructured_1.inp",
                              mesh_file)

mesh = P4estMesh{3}(mesh_file, polydeg=3,
                    mapping=mapping,
                    initial_refinement_level=1)

# TODO P4EST FSP
# # Refine bottom left quadrant of each tree to level 2
# function refine_fn(p8est, which_tree, quadrant)
#   if quadrant.x == 0 && quadrant.y == 0 && quadrant.z == 0 && quadrant.level < 2 && convert(Int, which_tree) == 0
#     # return true (refine)
#     return Cint(1)
#   else
#     # return false (don't refine)
#     return Cint(0)
#   end
# end

# Refine recursively until each bottom left quadrant of a tree has level 2
# The mesh will be rebalanced before the simulation starts
# refine_fn_c = @cfunction(refine_fn, Cint, (Ptr{Trixi.p8est_t}, Ptr{Trixi.p4est_topidx_t}, Ptr{Trixi.p8est_quadrant_t}))
# Trixi.refine_p4est!(mesh.p4est, true, refine_fn_c, C_NULL)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        # save_restart, save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), #maxiters=1,
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
