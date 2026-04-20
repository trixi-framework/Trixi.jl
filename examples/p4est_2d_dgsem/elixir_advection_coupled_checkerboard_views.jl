using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi
using P4est

###############################################################################
# Coupled mesh views with non-rectangular (checkerboard) geometry
#
# The parent mesh is split into two non-rectangular mesh views based on the
# sign of x*y.  Elements in quadrants I and III (x*y >= 0) form one view;
# elements in quadrants II and IV (x*y < 0) form the other.
#
#   y
#   ^
#   |  view 2 | view 1
#   |  (QII)  | (QI)
#   +---------+-------> x
#   |  view 1 | view 2
#   |  (QIII) | (QIV)
#
# Neither view is a rectangle, so every face direction (:x_neg, :x_pos,
# :y_neg, :y_pos) can appear at both the view interface and a physical domain
# edge within the same mesh view.  BoundaryConditionCoupledP4est is therefore
# assigned to all four names, with a Dirichlet fallback for the physical edges.
#
# Non-uniform p4est refinement creates hanging nodes (mortars) that may cross
# the view interface, exercising the coupled-mortar machinery.

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
# Refine a selection of trees to create internal hanging nodes; after 2:1
# balancing these hanging nodes will also appear at the checkerboard interface.
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

# Refine trees near the origin to concentrate resolution at the checkerboard
# interface and ensure hanging nodes cross it.
function refine_selected_trees_checkerboard(p4est_ptr, which_tree, quadrant_ptr)
    tree_id = which_tree + 1  # p4est is 0-indexed
    return tree_id in (1, 3, 5, 6, 9, 10, 13, 14) ? Cint(1) : Cint(0)
end

refine_fn_c = @cfunction(refine_selected_trees_checkerboard, Cint,
                         (Ptr{Trixi.p4est_t}, Trixi.p4est_topidx_t,
                          Ptr{Trixi.p4est_quadrant_t}))
init_fn_c = @cfunction(Trixi.init_fn, Cvoid,
                       (Ptr{Trixi.p4est_t}, Trixi.p4est_topidx_t,
                        Ptr{Trixi.p4est_quadrant_t}))

println("Refining selected trees...")
Trixi.refine_p4est!(parent_mesh.p4est, false, refine_fn_c, init_fn_c)

println("Balancing mesh (enforces 2:1 constraint)...")
Trixi.balance!(parent_mesh, init_fn_c)
Trixi.update_ghost_layer!(parent_mesh)

n_cells_after = Trixi.ncells(parent_mesh)
println("After refinement + balance: $(n_cells_after) elements")

# Build parent cache to access element coordinates
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
# STEP 2: Split into two non-rectangular mesh views
#
# View 1: quadrants I and III  (x*y >= 0)
# View 2: quadrants II and IV  (x*y < 0)

println("\n" * "="^80)
println("STEP 2: Checkerboard split into two non-rectangular mesh views")
println("="^80)

view1_elements = Int[]  # QI ∪ QIII
view2_elements = Int[]  # QII ∪ QIV

for element in 1:n_cells_after
    x_c = cache_parent.elements.node_coordinates[1, 2, 2, element]
    y_c = cache_parent.elements.node_coordinates[2, 2, 2, element]
    if x_c * y_c >= 0.0
        push!(view1_elements, element)
    else
        push!(view2_elements, element)
    end
end

println("Total elements : $n_cells_after")
println("View 1 (QI∪QIII, x*y≥0): $(length(view1_elements)) elements")
println("View 2 (QII∪QIV, x*y<0): $(length(view2_elements)) elements")

mesh1 = P4estMeshView(parent_mesh, view1_elements)
mesh2 = P4estMeshView(parent_mesh, view2_elements)

###############################################################################
# STEP 3: Boundary conditions
#
# Because neither mesh view is rectangular, every face direction (:x_neg,
# :x_pos, :y_neg, :y_pos) appears at both the view interface and a physical
# domain edge within the same view.  We therefore assign
# BoundaryConditionCoupledP4est to all four names and provide a Dirichlet
# fallback for the physical-domain faces (those with neighbor_ids_parent == 0).

println("\n" * "="^80)
println("STEP 3: Build boundary conditions")
println("="^80)

coupling_functions = Array{Function}(undef, 2, 2)
coupling_functions[1, 1] = (x, u, equations_other, equations_own) -> u
coupling_functions[1, 2] = (x, u, equations_other, equations_own) -> u
coupling_functions[2, 1] = (x, u, equations_other, equations_own) -> u
coupling_functions[2, 2] = (x, u, equations_other, equations_own) -> u

dirichlet_bc = BoundaryConditionDirichlet(initial_condition_convergence_test)

# fallback_bc = dirichlet_bc: physical-domain boundaries (neighbor_ids_parent == 0)
# fall back to Dirichlet; view-interface boundaries couple normally.
coupled_bc = BoundaryConditionCoupledP4est(coupling_functions; fallback_bc = dirichlet_bc)

# All four face directions can be view-interface faces in a non-rectangular split.
all_faces = Set([:x_neg, :x_pos, :y_neg, :y_pos])

function build_view_bcs(mesh_view, equations, solver, dirichlet_bc, coupled_bc,
                        view_interface_names)
    cache_temp = Trixi.create_cache(mesh_view, equations, solver,
                                    initial_condition_convergence_test, Float64)
    actual_names = unique(cache_temp.boundaries.name)
    sorted_names = Tuple(sort(collect(actual_names)))
    bc_values = Tuple(name in view_interface_names ? coupled_bc : dirichlet_bc
                      for name in sorted_names)
    bc_nt = NamedTuple{sorted_names}(bc_values)
    println("  Boundary names: $actual_names → $(Dict(n => (n in view_interface_names ? "Coupled(+fallback)" : "Dirichlet") for n in actual_names))")
    return bc_nt
end

println("Building view 1 BCs...")
bc1 = build_view_bcs(mesh1, equations, solver, dirichlet_bc, coupled_bc, all_faces)
println("Building view 2 BCs...")
bc2 = build_view_bcs(mesh2, equations, solver, dirichlet_bc, coupled_bc, all_faces)

semi1 = SemidiscretizationHyperbolic(mesh1, equations, initial_condition_convergence_test,
                                     solver, boundary_conditions = bc1)
semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition_convergence_test,
                                     solver, boundary_conditions = bc2)

semi = SemidiscretizationCoupledP4est(semi1, semi2; coupling_functions = coupling_functions)
println("="^80)

###############################################################################
# STEP 4: Mortar analysis

println("\n" * "="^80)
println("STEP 4: Mortar analysis")
println("="^80)

total_coupled_mortars = 0
for (i, semi_local) in enumerate(semi.semis)
    n_regular = Trixi.nmortars(semi_local.cache.mortars)
    n_coupled = Trixi.ncoupledmortars(semi_local.cache.coupled_mortars)
    global total_coupled_mortars += n_coupled
    println("Mesh view $i ($(length(semi_local.mesh.cell_ids)) elements):")
    println("  Regular mortars (internal hanging nodes): $n_regular")
    println("  Coupled mortars (hanging nodes at view boundary): $n_coupled")
end

if total_coupled_mortars > 0
    println("\nSUCCESS! $total_coupled_mortars coupled mortars with hanging nodes!")
else
    println("\nNo coupled mortars at view boundary.")
end
println("="^80)

###############################################################################
# STEP 5: Run simulation

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
println("Demonstrated coupled mesh views with non-rectangular (checkerboard) geometry!")
println("="^80)
