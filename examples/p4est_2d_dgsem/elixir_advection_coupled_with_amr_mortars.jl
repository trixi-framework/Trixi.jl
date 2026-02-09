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

# Global variable to track which trees to refine
const TREES_TO_REFINE = Set([1, 3, 5, 6, 9, 10, 13, 14])

# Custom refine function - refines quadrants in specified trees
function refine_left_half(p4est_ptr, which_tree, quadrant_ptr)
    # which_tree is 0-indexed in p4est
    tree_id = which_tree + 1
    if tree_id in TREES_TO_REFINE
        return Cint(1)  # refine
    else
        return Cint(0)  # don't refine
    end
end

# Create C-callable function pointers
refine_fn_c = @cfunction(refine_left_half, Cint,
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

# Use coupled boundary condition for all boundaries
coupled_bc = BoundaryConditionCoupledP4est(coupling_functions)

# Determine which domain boundaries each view touches based on element coordinates
function get_domain_boundaries(element_ids, elements_cache, domain_min, domain_max; tol=1e-10)
    boundaries = Set{Symbol}()
    for elem in element_ids
        coords = elements_cache.node_coordinates[:, :, :, elem]
        x_min, x_max = extrema(coords[1, :, :])
        y_min, y_max = extrema(coords[2, :, :])

        abs(x_min - domain_min[1]) < tol && push!(boundaries, :x_neg)
        abs(x_max - domain_max[1]) < tol && push!(boundaries, :x_pos)
        abs(y_min - domain_min[2]) < tol && push!(boundaries, :y_neg)
        abs(y_max - domain_max[2]) < tol && push!(boundaries, :y_pos)
    end
    return boundaries
end

# Get boundaries for each view
boundaries1 = get_domain_boundaries(left_elements, cache_parent.elements, coordinates_min, coordinates_max)
boundaries2 = get_domain_boundaries(right_elements, cache_parent.elements, coordinates_min, coordinates_max)

println("Left view boundaries: $boundaries1")
println("Right view boundaries: $boundaries2")

# Create boundary conditions for each view (coupled BC for all boundaries)
bc1 = Dict(name => coupled_bc for name in boundaries1)
bc2 = Dict(name => coupled_bc for name in boundaries2)

# Create semidiscretizations
semi1 = SemidiscretizationHyperbolic(mesh1, equations, initial_condition_convergence_test,
                                     solver, boundary_conditions = bc1)
semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition_convergence_test,
                                     solver, boundary_conditions = bc2)

# Create coupled system
semi = SemidiscretizationCoupledP4est(semi1, semi2)

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
