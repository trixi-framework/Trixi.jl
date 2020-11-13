
# TODO: Taal refactor, allow saving arbitrary functions of the conservative variables
# TODO: Taal refactor, make solution_variables a function instead of a Symbol
"""
    SaveSolutionCallback(; interval=0,
                           save_initial_solution=true,
                           save_final_solution=true,
                           output_directory="out",
                           solution_variables=:primitive)

Save the current numerical solution every `interval` time steps.
"""
mutable struct SaveSolutionCallback
  interval::Int
  save_initial_solution::Bool
  save_final_solution::Bool
  output_directory::String
  solution_variables::Symbol
end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:SaveSolutionCallback}
  save_solution_callback = cb.affect!
  print(io, "SaveSolutionCallback(interval=", save_solution_callback.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:SaveSolutionCallback}
  if get(io, :compact, false)
    show(io, cb)
  else
    save_solution_callback = cb.affect!

    key_width = get(io, :key_width, 25)
    total_width = get(io, :total_width, 100)
    setup = [ 
             "interval" => save_solution_callback.interval,
             "solution variables" => save_solution_callback.solution_variables,
             "save initial solution" => save_solution_callback.save_initial_solution ? "yes" : "no",
             "save final solution" => save_solution_callback.save_final_solution ? "yes" : "no",
             "output directory" => abspath(normpath(save_solution_callback.output_directory)),
            ]
    print(io, boxed_setup("SaveSolutionCallback", key_width, total_width, setup))
  end
end


function SaveSolutionCallback(; interval=0,
                                save_initial_solution=true,
                                save_final_solution=true,
                                output_directory="out",
                                solution_variables=:primitive)
  # when is the callback activated
  condition = (u, t, integrator) -> interval > 0 && ((integrator.iter % interval == 0) ||
                                                     (save_final_solution && isfinished(integrator)))

  solution_callback = SaveSolutionCallback(interval, save_initial_solution, save_final_solution,
                                           output_directory, solution_variables)

  DiscreteCallback(condition, solution_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:SaveSolutionCallback}
  solution_callback = cb.affect!

  mpi_isroot() && mkpath(solution_callback.output_directory)

  semi = integrator.p
  mesh, _, _, _ = mesh_equations_solver_cache(semi)
  @timeit_debug timer() "I/O" begin
    if mesh.unsaved_changes
      mesh.current_filename = save_mesh_file(mesh, solution_callback.output_directory)
      mesh.unsaved_changes = false
    end
  end

  if solution_callback.save_initial_solution
    solution_callback(integrator)
  end

  return nothing
end


function (solution_callback::SaveSolutionCallback)(integrator)
  u_ode = integrator.u
  @unpack t, dt, iter = integrator
  semi = integrator.p
  mesh, _, _, _ = mesh_equations_solver_cache(semi)

  @timeit_debug timer() "I/O" begin
    if mesh.unsaved_changes
      mesh.current_filename = save_mesh_file(mesh, solution_callback.output_directory, iter)
      mesh.unsaved_changes = false
    end

    element_variables = Dict{Symbol, Any}()
    get_element_variables!(element_variables, u_ode, semi)
    callbacks = integrator.opts.callback
    if callbacks isa CallbackSet
      for cb in callbacks.continuous_callbacks
        get_element_variables!(element_variables, u_ode, semi, cb; t=integrator.t, iter=integrator.iter)
      end
      for cb in callbacks.discrete_callbacks
        get_element_variables!(element_variables, u_ode, semi, cb; t=integrator.t, iter=integrator.iter)
      end
    end

    save_solution_file(u_ode, t, dt, iter, semi, solution_callback, element_variables)
  end

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)
  return nothing
end


@inline function save_solution_file(u_ode::AbstractVector, t, dt, iter,
                                    semi::AbstractSemidiscretization, solution_callback,
                                    element_variables=Dict{Symbol,Any}())
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  u = wrap_array(u_ode, mesh, equations, solver, cache)
  save_solution_file(u, t, dt, iter, mesh, equations, solver, cache, solution_callback, element_variables)
end


# TODO: Taal refactor, move save_mesh_file?
# function save_mesh_file(mesh::TreeMesh, output_directory, timestep=-1) in io/io.jl

include("save_solution_dg.jl")
