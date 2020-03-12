module AMR

using ..Trixi
using ..Auxiliary: parameter, timer
using ..Auxiliary.Containers: append!
using ..Mesh: TreeMesh
using ..Mesh.Trees: Tree, refine!, refine_box!, coarsen_box!
using ..Solvers: AbstractSolver

using TimerOutputs: @timeit, print_timer
using HDF5: h5open, attrs


function adapt!(mesh::TreeMesh, solver::AbstractSolver)
  @show "wololo!"
  return true
end


end # module AMR
