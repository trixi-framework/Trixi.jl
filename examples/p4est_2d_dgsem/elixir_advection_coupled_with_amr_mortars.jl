using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi
using P4est

###############################################################################
# Coupled mesh views with ACTUAL hanging nodes
#
# This example creates non-uniform refinement by directly calling p4est
# refinement on specific trees, creating hanging nodes (mortars) in the mesh.
# The mesh views are then created to cross these mortar interfaces.

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)

# Create parent mesh with 4x4 trees, NO initial refinement
# We'll manually refine some trees to create hanging nodes
trees_per_dimension = (4, 4)
parent_mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                        coordinates_min = coordinates_min,
                        coordinates_max = coordinates_max,
                        initial_refinement_level = 0,
                        periodicity = false)

println("\n" * "="^80)
println("STEP 1: Create non-uniform refinement")
println("="^80)
println("Initial mesh: $(Trixi.ncells(parent_mesh)) elements (trees)")

# Define a custom refine callback that refines selected trees
# Trees are numbered 1-16 in a 4x4 grid, row by row:
# 13 14 15 16
#  9 10 11 12
#  5  6  7  8
#  1  2  3  4
#
# We refine trees that create hanging nodes crossing the x=0 boundary:
# - Trees 1, 5, 6, 9, 10, 13, 14 on the left (x < 0)
# - Tree 3 on the right (x > 0)
# This creates coupled mortars where BOTH views have hanging nodes at the interface.

# Custom refine function - refines quadrants in specified trees.
# Trees are inlined as a tuple so no global constant is needed.
function refine_selected_trees(p4est_ptr, which_tree, quadrant_ptr)
    tree_id = which_tree + 1  # p4est is 0-indexed
    return tree_id in (1, 3, 5, 6, 9, 10, 13, 14) ? Cint(1) : Cint(0)
end

# Create C-callable function pointers
refine_fn_c = @cfunction(refine_selected_trees, Cint,
                         (Ptr{Trixi.p4est_t}, Trixi.p4est_topidx_t,
                          Ptr{Trixi.p4est_quadrant_t}))
init_fn_c = @cfunction(Trixi.init_fn, Cvoid,
                       (Ptr{Trixi.p4est_t}, Trixi.p4est_topidx_t,
                        Ptr{Trixi.p4est_quadrant_t}))

# Refine the mesh (non-recursive, just one level)
println("Refining selected trees (1,3,5,6,9,10,13,14)...")
Trixi.refine_p4est!(parent_mesh.p4est, false, refine_fn_c, init_fn_c)

# Balance to ensure 2:1 constraint (this creates additional mortars)
println("Balancing mesh...")
Trixi.balance!(parent_mesh, init_fn_c)

# Update ghost layer
Trixi.update_ghost_layer!(parent_mesh)

n_cells_after = Trixi.ncells(parent_mesh)
println("After refinement: $(n_cells_after) elements")

# IMPORTANT: Create a temporary semidiscretization on the parent mesh to build the cache
# This is required before creating mesh views
println("Building parent mesh cache...")
boundary_conditions_parent = Dict(:x_neg => BoundaryConditionDirichlet(initial_condition_convergence_test),
                                  :x_pos => BoundaryConditionDirichlet(initial_condition_convergence_test),
                                  :y_neg => BoundaryConditionDirichlet(initial_condition_convergence_test),
                                  :y_pos => BoundaryConditionDirichlet(initial_condition_convergence_test))
semi_parent = SemidiscretizationHyperbolic(parent_mesh, equations,
                                           initial_condition_convergence_test,
                                           solver,
                                           boundary_conditions = boundary_conditions_parent)
cache_parent = semi_parent.cache
println("Parent mesh has $(Trixi.nmortars(cache_parent.mortars)) mortars (hanging nodes)")
println("="^80)

###############################################################################
# STEP 2: Create mesh views

println("\n" * "="^80)
println("STEP 2: Create mesh views from non-uniformly refined mesh")
println("="^80)

# Now we can determine element positions from the cache
total_elements = n_cells_after

# Split by x-coordinate: elements with center x < 0 go to left view
left_elements = Int[]
right_elements = Int[]

for element in 1:total_elements
    x_center = cache_parent.elements.node_coordinates[1, 2, 2, element]
    if x_center < 0.0
        push!(left_elements, element)
    else
        push!(right_elements, element)
    end
end

println("Total elements: $total_elements")
println("Left view: $(length(left_elements)) elements")
println("Right view: $(length(right_elements)) elements")

# Create mesh views
mesh1 = P4estMeshView(parent_mesh, left_elements)
mesh2 = P4estMeshView(parent_mesh, right_elements)

# Define coupling functions (identity for same equation)
coupling_functions = Array{Function}(undef, 2, 2)
coupling_functions[1, 1] = (x, u, equations_other, equations_own) -> u
coupling_functions[1, 2] = (x, u, equations_other, equations_own) -> u
coupling_functions[2, 1] = (x, u, equations_other, equations_own) -> u
coupling_functions[2, 2] = (x, u, equations_other, equations_own) -> u

# Dirichlet BC for physical domain boundaries
dirichlet_bc = BoundaryConditionDirichlet(initial_condition_convergence_test)

# Coupled BC for view interfaces.
# For simple axis-aligned splits (x-split or y-split), view_interface_names can be
# set to only the face names that appear exclusively at the view interface, and
# fallback_bc can be omitted (defaults to nothing, which errors on unexpected zeros).
#
# For mixed-geometry splits (e.g. diagonal), the same face name may appear at both
# a view interface and a physical domain edge within the same mesh view.  In that
# case pass fallback_bc = dirichlet_bc and set view_interface_names to all four
# face directions.  Boundaries with no coupling neighbor (neighbor_ids_parent == 0)
# then fall back to the Dirichlet BC automatically.
coupled_bc = BoundaryConditionCoupledP4est(coupling_functions; fallback_bc = dirichlet_bc)

# View interface names for the x=0 split (left-right):
#   left view's x_pos face and right view's x_neg face are at x=0.
# For a y-split use Set([:y_pos]) / Set([:y_neg]).
# For a mixed split where every face direction can appear at the view interface,
# use Set([:x_neg, :x_pos, :y_neg, :y_pos]) for both views (requires fallback_bc above).
view_interface_names_left = Set([:x_pos])   # left view's x_pos is the view interface
view_interface_names_right = Set([:x_neg])  # right view's x_neg is the view interface

function build_view_bcs(mesh_view, equations, solver,
                        dirichlet_bc, coupled_bc, view_interface_names)
    # Call create_cache to discover actual boundary names without building a full semi
    cache_temp = Trixi.create_cache(mesh_view, equations, solver,
                                    initial_condition_convergence_test, Float64)
    actual_names = unique(cache_temp.boundaries.name)

    # Build as a NamedTuple so digest_boundary_conditions routes to
    # UnstructuredSortedBoundaryTypes (the Dict path would be treated as a
    # single BC value and silently expanded to all four sides).
    sorted_names = Tuple(sort(collect(actual_names)))
    bc_values = Tuple(name in view_interface_names ? coupled_bc : dirichlet_bc
                      for name in sorted_names)
    bc_nt = NamedTuple{sorted_names}(bc_values)
    println("  Boundary names: $actual_names → $(Dict(n => (n in view_interface_names ? "Coupled" : "Dirichlet") for n in actual_names))")
    return bc_nt
end

println("Building left view BCs...")
bc1 = build_view_bcs(mesh1, equations, solver, dirichlet_bc, coupled_bc, view_interface_names_left)
println("Building right view BCs...")
bc2 = build_view_bcs(mesh2, equations, solver, dirichlet_bc, coupled_bc, view_interface_names_right)

# Create semidiscretizations
semi1 = SemidiscretizationHyperbolic(mesh1, equations, initial_condition_convergence_test,
                                     solver, boundary_conditions = bc1)
semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition_convergence_test,
                                     solver, boundary_conditions = bc2)

# Create coupled system
semi = SemidiscretizationCoupledP4est(semi1, semi2; coupling_functions = coupling_functions)

println("="^80)

###############################################################################
# STEP 3: Check for mortars

println("\n" * "="^80)
println("STEP 3: Mortar analysis")
println("="^80)

total_coupled_mortars = 0
for (i, semi_local) in enumerate(semi.semis)
    mesh_local = semi_local.mesh
    cache_local = semi_local.cache

    n_regular = Trixi.nmortars(cache_local.mortars)
    n_coupled = Trixi.ncoupledmortars(cache_local.coupled_mortars)
    global total_coupled_mortars += n_coupled

    println("Mesh view $i:")
    println("  Elements: $(length(mesh_local.cell_ids))")
    println("  Regular mortars (internal hanging nodes): $n_regular")
    println("  Coupled mortars (hanging nodes at view boundary): $n_coupled")
    println()
end

if total_coupled_mortars > 0
    println("SUCCESS! $total_coupled_mortars coupled mortars with hanging nodes!")
else
    println("No coupled mortars at view boundary.")
    println("Hanging nodes exist but are all within individual views.")
end
println("="^80)

###############################################################################
# STEP 4: Run simulation

println("\nRunning simulation...")

ode = semidiscretize(semi, (0.0, 20.0))

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

println("\n" * "="^80)
println("SIMULATION COMPLETED")
println("="^80)
println("Successfully demonstrated coupled mesh views with non-uniform refinement!")
println("="^80)
