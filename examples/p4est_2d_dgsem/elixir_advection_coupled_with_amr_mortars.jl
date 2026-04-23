using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi
using P4est

###############################################################################
# Coupled mesh views with hanging nodes (mortars) at the view interface.
#
# A 4x4 parent mesh is non-uniformly refined by directly calling p4est,
# creating hanging nodes in the mesh.  The parent mesh is then split at x=0
# into two P4estMeshViews.  Because trees on both sides of x=0 are refined,
# the view interface crosses mortar boundaries, exercising the coupled-mortar
# machinery.

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)

# 4x4 tree grid — tree numbering (row-major from bottom-left):
#   13 14 15 16   (y ∈ [0.5, 1])
#    9 10 11 12   (y ∈ [0,   0.5])
#    5  6  7  8   (y ∈ [-0.5, 0])
#    1  2  3  4   (y ∈ [-1, -0.5])
#   col: 1  2  3  4   (x: -1→-0.5, -0.5→0, 0→0.5, 0.5→1)
#
# Trees 1,5,6,9,10,13,14 are on the left (x < 0); tree 3 is on the right.
# Refining both sides of x=0 ensures coupled mortars exist at the interface.
trees_per_dimension = (4, 4)
parent_mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                        coordinates_min = coordinates_min,
                        coordinates_max = coordinates_max,
                        initial_refinement_level = 0,
                        periodicity = false)

function refine_selected_trees(p4est_ptr, which_tree, quadrant_ptr)
    tree_id = which_tree + 1  # p4est is 0-indexed
    return tree_id in (1, 3, 5, 6, 9, 10, 13, 14) ? Cint(1) : Cint(0)
end

refine_fn_c = @cfunction(refine_selected_trees, Cint,
                         (Ptr{Trixi.p4est_t}, Trixi.p4est_topidx_t,
                          Ptr{Trixi.p4est_quadrant_t}))
init_fn_c = @cfunction(Trixi.init_fn, Cvoid,
                       (Ptr{Trixi.p4est_t}, Trixi.p4est_topidx_t,
                        Ptr{Trixi.p4est_quadrant_t}))

# Refine selected trees (non-recursive, one level) then balance for 2:1 constraint.
Trixi.refine_p4est!(parent_mesh.p4est, false, refine_fn_c, init_fn_c)
Trixi.balance!(parent_mesh, init_fn_c)
Trixi.update_ghost_layer!(parent_mesh)

###############################################################################
# Build a temporary semidiscretization on the parent mesh to obtain element
# coordinates needed for splitting elements between the two views.

boundary_conditions_parent = Dict(:x_neg => BoundaryConditionDirichlet(initial_condition_convergence_test),
                                  :x_pos => BoundaryConditionDirichlet(initial_condition_convergence_test),
                                  :y_neg => BoundaryConditionDirichlet(initial_condition_convergence_test),
                                  :y_pos => BoundaryConditionDirichlet(initial_condition_convergence_test))
semi_parent = SemidiscretizationHyperbolic(parent_mesh, equations,
                                           initial_condition_convergence_test,
                                           solver,
                                           boundary_conditions = boundary_conditions_parent)
cache_parent = semi_parent.cache

###############################################################################
# Split elements at x=0: left view (x < 0) and right view (x ≥ 0).

left_elements = Int[]
right_elements = Int[]

for element in 1:Trixi.ncells(parent_mesh)
    x_center = cache_parent.elements.node_coordinates[1, 2, 2, element]
    if x_center < 0.0
        push!(left_elements, element)
    else
        push!(right_elements, element)
    end
end

mesh1 = P4estMeshView(parent_mesh, left_elements)
mesh2 = P4estMeshView(parent_mesh, right_elements)

###############################################################################
# Boundary conditions.
#
# The split is axis-aligned at x=0, so :x_pos appears only at the view
# interface in the left view and :x_neg only in the right view.  All other
# face names correspond exclusively to physical domain boundaries.
#
# fallback_bc is provided so that the same elixir pattern also works for
# non-rectangular splits where a face name may appear at both the view
# interface and a physical domain edge (see elixir_advection_coupled_checkerboard_views.jl).

coupling_functions = Array{Function}(undef, 2, 2)
coupling_functions[1, 1] = (x, u, equations_other, equations_own) -> u
coupling_functions[1, 2] = (x, u, equations_other, equations_own) -> u
coupling_functions[2, 1] = (x, u, equations_other, equations_own) -> u
coupling_functions[2, 2] = (x, u, equations_other, equations_own) -> u

dirichlet_bc = BoundaryConditionDirichlet(initial_condition_convergence_test)
coupled_bc = BoundaryConditionCoupledP4est(coupling_functions; fallback_bc = dirichlet_bc)

view_interface_names_left = Set([:x_pos])
view_interface_names_right = Set([:x_neg])

function build_view_bcs(mesh_view, equations, solver,
                        dirichlet_bc, coupled_bc, view_interface_names)
    cache_temp = Trixi.create_cache(mesh_view, equations, solver,
                                    initial_condition_convergence_test, Float64)
    actual_names = unique(cache_temp.boundaries.name)
    sorted_names = Tuple(sort(collect(actual_names)))
    bc_values = Tuple(name in view_interface_names ? coupled_bc : dirichlet_bc
                      for name in sorted_names)
    return NamedTuple{sorted_names}(bc_values)
end

bc1 = build_view_bcs(mesh1, equations, solver, dirichlet_bc, coupled_bc,
                     view_interface_names_left)
bc2 = build_view_bcs(mesh2, equations, solver, dirichlet_bc, coupled_bc,
                     view_interface_names_right)

semi1 = SemidiscretizationHyperbolic(mesh1, equations, initial_condition_convergence_test,
                                     solver, boundary_conditions = bc1)
semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition_convergence_test,
                                     solver, boundary_conditions = bc2)

semi = SemidiscretizationCoupledP4est(semi1, semi2; coupling_functions = coupling_functions)

###############################################################################
# ODE problem and callbacks.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback1 = AnalysisCallback(semi1, interval = 50)
analysis_callback2 = AnalysisCallback(semi2, interval = 50)
analysis_callback = AnalysisCallbackCoupledP4est(semi, analysis_callback1,
                                                 analysis_callback2)

save_solution = SaveSolutionCallback(interval = 1,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.6)

callbacks = CallbackSet(summary_callback, analysis_callback,
                        save_solution, stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0,
            save_everystep = false,
            ode_default_options()..., callback = callbacks)

summary_callback()
