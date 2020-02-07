#!/usr/bin/env julia

include("../src/mesh/trees.jl")

using .Trees: Tree, refine!, size, capacity, leaf_nodes, refine_box!
using TimerOutputs
using Profile

to = TimerOutput()

x_start = -16
x_end = 16
capacity_ = 1000
initial_refinement_level = 2

domain_center = (x_start + x_end) / 2
domain_length = x_end - x_start

# Create tree object
@timeit to "create tree" t = Tree(Val{1}(), capacity_, [domain_center], domain_length)
println("Initial tree:")
println(t)

# Create initial refinement
@timeit to "initial refinement" for l = 1:initial_refinement_level
  @timeit to "refine!" refine!(t)
  println("After uniform refinement to level $l:")
  println(t)
end

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

left_nodes = refine_box!(t, -16, 0)
right_nodes = refine_box!(t, 0, 16)
center_nodes = refine_box!(t, -8, 8)

# Print tree information
@show leaf_nodes(t)
@show t.coordinates[:, leaf_nodes(t)]
@show left_nodes
@show right_nodes
@show center_nodes

print_timer(to)
println()
