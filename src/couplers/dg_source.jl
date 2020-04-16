module DgSourceCoupler

using ...Trixi
using ..Couplers # Use everything to allow method extension via "function <parent_module>.<method>"
import ...Solvers
using ...Solvers.DgSolver: polydeg
#=using ...Equations: AbstractEquation, initial_conditions, calcflux!, calcflux_twopoint!,=#
#=                    riemann!, sources, calc_max_dt, cons2entropy, cons2indicator!, cons2prim=#
#=import ...Equations: nvariables # Import to allow method extension=#
using ...Auxiliary: timer, parameter
using ...Mesh: TreeMesh
#=using ...Mesh.Trees: leaf_cells, length_at_cell, n_directions, has_neighbor,=#
#=                     opposite_direction, has_coarse_neighbor, has_child, has_children=#

using TimerOutputs: @timeit, @notimeit

export DgSource
export couple_post_substep


# Main coupler data structure
mutable struct DgSource{SolverA, SolverB, NA, NB} <: AbstractCoupler
  solver_a::SolverA
  solver_b::SolverB
end


# Convenience constructor to create DG solver instance
function DgSource(solver_a::SolverA, solver_b::SolverB, mesh::TreeMesh) where {SolverA, SolverB}
  # Create actual coupler instance
  coupler = DgSource{SolverA, SolverB, polydeg(solver_a), polydeg(solver_b)}(solver_a, solver_b)

  return coupler
end

end # module
