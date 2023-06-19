using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_convergence_test

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(
  :all => boundary_condition
)

# Solver with polydeg=4 to ensure free stream preservation (FSP) on non-conforming meshes.
# The polydeg of the solver must be at least twice as big as the polydeg of the mesh.
# See https://doi.org/10.1007/s10915-018-00897-9, Section 6.
solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs,
               volume_integral=VolumeIntegralWeakForm())

# Mapping as described in https://arxiv.org/abs/2012.12040 but with less warping.
# The mapping will be interpolated at tree level, and then refined without changing
# the geometry interpolant. The original mapping applied to this unstructured mesh
# causes some Jacobians to be negative, which makes the mesh invalid.
function mapping(xi, eta, zeta)
  # Don't transform input variables between -1 and 1 onto [0,3] to obtain curved boundaries
  # xi = 1.5 * xi_ + 1.5
  # eta = 1.5 * eta_ + 1.5
  # zeta = 1.5 * zeta_ + 1.5

  y = eta + 1/6 * (cos(1.5 * pi * (2 * xi - 3)/3) *
                   cos(0.5 * pi * (2 * eta - 3)/3) *
                   cos(0.5 * pi * (2 * zeta - 3)/3))

  x = xi + 1/6 * (cos(0.5 * pi * (2 * xi - 3)/3) *
                  cos(2 * pi * (2 * y - 3)/3) *
                  cos(0.5 * pi * (2 * zeta - 3)/3))

  z = zeta + 1/6 * (cos(0.5 * pi * (2 * x - 3)/3) *
                    cos(pi * (2 * y - 3)/3) *
                    cos(0.5 * pi * (2 * zeta - 3)/3))

  # Transform the weird deformed cube to be approximately the cube [0,2]^3
  return SVector(x + 1, y + 1, z + 1)
end

# Unstructured mesh with 68 cells of the cube domain [-1, 1]^3
mesh_file = joinpath(@__DIR__, "cube_unstructured_1.inp")
isfile(mesh_file) || download("https://gist.githubusercontent.com/efaulhaber/d45c8ac1e248618885fa7cc31a50ab40/raw/37fba24890ab37cfa49c39eae98b44faf4502882/cube_unstructured_1.inp",
                              mesh_file)

# INP mesh files are only support by p4est. Hence, we
# create a p4est connecvity object first from which
# we can create a t8code mesh.
conn = Trixi.read_inp_p4est(mesh_file,Val(3))

# Mesh polydeg of 2 (half the solver polydeg) to ensure FSP (see above).
mesh = T8codeMesh{3}(conn, polydeg=2,
                    mapping=mapping,
                    initial_refinement_level=0)

# Note: This is actually a `p8est_quadrant_t` which is much bigger than the
# following struct. But we only need the first four fields for our purpose.
struct t8_dhex_t
  x :: Int32
  y :: Int32
  z :: Int32
  level :: Int8
  # [...] # See `p8est.h` in `p4est` for more info.
end

function adapt_callback(forest,
                        forest_from,
                        which_tree,
                        lelement_id,
                        ts,
                        is_family, 
                        num_elements,
                        elements_ptr) :: Cint

  vertex = Vector{Cdouble}(undef,3)
  elements = unsafe_wrap(Array, elements_ptr, num_elements)
  Trixi.t8_element_vertex_reference_coords(ts, elements[1], 0, pointer(vertex))

  el = unsafe_load(Ptr{t8_dhex_t}(elements[1]))

  if el.x == 0 && el.y == 0 && el.z == 0 && el.level < 2
    # return true (refine)
    return 1
  else
    # return false (don't refine)
    return 0
  end
end

@assert(Trixi.t8_forest_is_committed(mesh.forest) != 0);

# Init new forest.
new_forest_ref = Ref{Trixi.t8_forest_t}()
Trixi.t8_forest_init(new_forest_ref);
new_forest = new_forest_ref[]

let set_from = C_NULL, recursive = 1, set_for_coarsening = 0, no_repartition = 0, do_ghost = 1
  Trixi.t8_forest_set_user_data(new_forest, C_NULL)
  Trixi.t8_forest_set_adapt(new_forest, mesh.forest, @Trixi.t8_adapt_callback(adapt_callback), recursive)
  Trixi.t8_forest_set_balance(new_forest, set_from, no_repartition)
  Trixi.t8_forest_set_partition(new_forest, set_from, set_for_coarsening)
  Trixi.t8_forest_set_ghost(new_forest, do_ghost, Trixi.T8_GHOST_FACES);
  Trixi.t8_forest_commit(new_forest)
end

mesh.forest = new_forest

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_convergence_test,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.045)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

# Not supported yet.
# save_solution = SaveSolutionCallback(interval=100,
#                                      save_initial_solution=true,
#                                      save_final_solution=true,
#                                      solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.6)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        # save_solution,
                        stepsize_callback);

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
