module DgSourceCoupler

using ...Trixi
using ..Couplers # Use everything to allow method extension via "function <parent_module>.<method>"
import ...Solvers
using ...Solvers.DgSolver: polydeg, nnodes, nvariables
using ...Auxiliary: timer, parameter
using ...Mesh: TreeMesh

using TimerOutputs: @timeit, @notimeit

export DgSource
export couple_post_rhs!


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


function Couplers.couple_post_rhs!(coupler::DgSource)
  dg_a = coupler.solver_a
  dg_b = coupler.solver_b

  @assert dg_a.n_elements == dg_b.n_elements "number of elements does not match"
  @assert nnodes(dg_a) == nnodes(dg_b) "number of nodes does not match"

  n_elements = dg_a.n_elements
  n_nodes = nnodes(dg_a)
  n_variables = nvariables(dg_a)

  for element_id in 1:n_elements
    for j in 1:n_nodes, i in 1:n_nodes
      for v in 1:n_variables
        # In combination with the "coupler_test_source", this essentially adds a numerical zero
        dg_a.elements.u_t[v, i, j, element_id] -= dg_b.elements.u[v, i, j, element_id]
        dg_b.elements.u_t[v, i, j, element_id] -= dg_a.elements.u[v, i, j, element_id]
      end
    end
  end
end

end # module
