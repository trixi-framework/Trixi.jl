
# TODO: Taal, implement, save AMR indicator values
# TODO: Taal, refactor, allow saving arbitrary functions of the conservative variables
mutable struct SaveSolutionCallback
  save_initial_solution::Bool
  save_final_solution::Bool
  output_directory::String
  solution_variables::Symbol
end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:SaveSolutionCallback}
  stepsize_callback = cb.affect!
  print(io, "SaveSolutionCallback")
end
# TODO: Taal bikeshedding, implement a method with more information and the signature
# function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:StepsizeCallback}
# end


function SaveSolutionCallback(; solution_interval=0,
                                save_initial_solution=true,
                                save_final_solution=true,
                                output_directory="out",
                                solution_variables=:primitive)
  condition = (u, t, integrator) -> solution_interval > 0 && ((integrator.iter % solution_interval == 0) || (save_final_solution && t == integrator.sol.prob.tspan[2]))

  solution_callback = SaveSolutionCallback(save_initial_solution, save_final_solution,
                                           output_directory, solution_variables)

  DiscreteCallback(condition, solution_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:SaveSolutionCallback}
  reset_timer!(timer())
  solution_callback = cb.affect!

  mkpath(solution_callback.output_directory)

  semi = integrator.p
  @unpack mesh = semi
  mesh.unsaved_changes = true

  if solution_callback.save_initial_solution
    solution_callback(integrator)
  end

  return nothing
end


function (solution_callback::SaveSolutionCallback)(integrator)
  @unpack u, t, dt, iter = integrator
  semi = integrator.p
  @unpack mesh, equations, solver, cache = semi

  @timeit_debug timer() "I/O" begin
    if mesh.unsaved_changes
      mesh.current_filename = save_mesh_file(mesh, solution_callback.output_directory, iter)
      mesh.unsaved_changes = false
    end

    save_solution_file(u, t, dt, iter, mesh, equations, solver, cache, solution_callback)
  end

  return nothing
end


# function save_mesh_file(mesh::TreeMesh, output_directory, timestep=-1) in io/io.jl
#

include("save_solution_dg.jl")
