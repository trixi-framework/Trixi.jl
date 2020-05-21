#!/usr/bin/env julia

include("../src/mesh/trees.jl")

using .Trees: Tree, refine!, length, capacity, leaf_cells, refine_box!,
              minimum_level, maximum_level, count_leaf_cells
using TimerOutputs
using Profile
using HDF5: h5open, attrs

const ndim = 2

# Save current mesh with some context information as an HDF5 file.
function save_mesh_file(filename::String, tree::Tree)
  # Open file (clobber existing content)
  h5open(filename, "w") do file
    # Add context information as attributes
    n_cells = length(tree)
    attrs(file)["ndim"] = ndim
    attrs(file)["n_cells"] = n_cells
    attrs(file)["n_leaf_cells"] = count_leaf_cells(tree)
    attrs(file)["minimum_level"] = minimum_level(tree)
    attrs(file)["maximum_level"] = maximum_level(tree)
    attrs(file)["center_level_0"] = tree.center_level_0
    attrs(file)["length_level_0"] = tree.length_level_0

    # Add tree data
    file["parent_ids"] = @view tree.parent_ids[1:n_cells]
    file["child_ids"] = @view tree.child_ids[:, 1:n_cells]
    file["neighbor_ids"] = @view tree.neighbor_ids[:, 1:n_cells]
    file["levels"] = @view tree.levels[1:n_cells]
    file["coordinates"] = @view tree.coordinates[:, 1:n_cells]
  end

  return filename * ".h5"
end

to = TimerOutput()

coordinates_min = [-16, -16]
coordinates_max = [ 16,  16]
capacity_ = 1000
initial_refinement_level = 2

domain_center = (coordinates_min + coordinates_max) / 2
domain_length = maximum(coordinates_max - coordinates_min)

# Create tree object
@timeit to "create tree" t = Tree(Val{2}(), capacity_, domain_center, domain_length)
println("Initial tree:")
println(t)

# Create initial refinement
@timeit to "initial refinement" for l = 1:initial_refinement_level
  @timeit to "refine!" refine!(t)
  println("After uniform refinement to level $l:")
  println(t)
end

# Add patches
patches = [
           [[0.0, -16.0], [16.0, 0.0]],
           [[0.0, -16.0], [16.0,  0.0]],
           [[0.0, -16.0], [16.0, -8.0]],
           #=[[8.0, -16.0], [16.0, -8.0]],=#
          ]
for (coordinates_min, coordinates_max) in patches
  refine_box!(t, coordinates_min, coordinates_max)
  println("After refinement patch $coordinates_min, $coordinates_max:")
  println(t)
end

save_mesh_file("mesh_test.h5", t)
exit(0)

# Create non-uniform refinement
@timeit to "local refinement" begin
  @timeit to "refine!" refine!(t, 4)
  println("After refining node 4:")
  println(t)
  @timeit to "refine!" refine!(t, 5)
  println("After refining node 5:")
  println(t)
end

# Refine one level further
@timeit to "refine!" refine!(t)
println("After refining everything once more:")
println(t)

refine_box!(t, -8, 8)
println("After refining center nodes:")
println(t)

# Print tree information
@show leaf_nodes(t)
@show t.coordinates[:, leaf_nodes(t)]

print_timer(to)
println()
