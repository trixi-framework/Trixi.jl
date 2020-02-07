#!/usr/bin/env julia

include("../src/mesh/trees.jl")

using .Trees: Tree, refine!, size, capacity, leaf_nodes
using TimerOutputs
using Profile

to = TimerOutput()

x_start = -16
x_end = 16
capacity_ = 1000
initial_refinement_level = 5

domain_center = (x_start + x_end) / 2
domain_length = x_end - x_start

# Create tree object
@timeit to "create tree" t = Tree(Val{1}(), capacity_, [domain_center], domain_length)
println(t)

# Create initial refinement
@timeit to "initial refinement" for l = 1:initial_refinement_level
  refine!(t)
  println(t)
end

# Print tree information
@show leaf_nodes(t)
@show t.coordinates[:, leaf_nodes(t)]

print_timer(to)
println()
