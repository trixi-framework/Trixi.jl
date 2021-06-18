
"""
    SaveSolutionCallback(; interval=0,
                           save_initial_solution=true,
                           save_final_solution=true,
                           output_directory="out",
                           solution_variables=cons2prim)

Save the current numerical solution every `interval` time steps. `solution_variables` can be any
callable that converts the conservative variables at a single point to a set of solution variables.
The first parameter passed to `solution_variables` will be the set of conservative variables and the
second parameter is the equation struct.
"""
mutable struct SaveSolutionCallback{SolutionVariables}
  interval::Int
  save_initial_solution::Bool
  save_final_solution::Bool
  output_directory::String
  solution_variables::SolutionVariables
end


function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SaveSolutionCallback})
  @nospecialize cb # reduce precompilation time

  save_solution_callback = cb.affect!
  print(io, "SaveSolutionCallback(interval=", save_solution_callback.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:SaveSolutionCallback})
  @nospecialize cb # reduce precompilation time

  if get(io, :compact, false)
    show(io, cb)
  else
    save_solution_callback = cb.affect!

    setup = [
             "interval" => save_solution_callback.interval,
             "solution variables" => save_solution_callback.solution_variables,
             "save initial solution" => save_solution_callback.save_initial_solution ? "yes" : "no",
             "save final solution" => save_solution_callback.save_final_solution ? "yes" : "no",
             "output directory" => abspath(normpath(save_solution_callback.output_directory)),
            ]
    summary_box(io, "SaveSolutionCallback", setup)
  end
end


function SaveSolutionCallback(; interval=0,
                                save_initial_solution=true,
                                save_final_solution=true,
                                output_directory="out",
                                solution_variables=cons2prim)

  # FIXME: Deprecations introduced in v0.3
  if solution_variables isa Symbol
    Base.depwarn("Providing the keyword argument `solution_variables` as a `Symbol` is deprecated." *
                 "Use functions such as `cons2cons` or `cons2prim` instead.", :SaveSolutionCallback)
    if solution_variables == :conservative
      solution_variables = cons2cons
    elseif solution_variables == :primitive
      solution_variables = cons2prim
    else
      error("Unknown `solution_variables` $solution_variables.")
    end
  end

  solution_callback = SaveSolutionCallback(interval, save_initial_solution, save_final_solution,
                                           output_directory, solution_variables)

  DiscreteCallback(solution_callback, solution_callback, # the first one is the condition, the second the affect!
                   save_positions=(false,false),
                   initialize=initialize!)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:SaveSolutionCallback}
  solution_callback = cb.affect!

  mpi_isroot() && mkpath(solution_callback.output_directory)

  semi = integrator.p
  mesh, _, _, _ = mesh_equations_solver_cache(semi)
  @trixi_timeit timer() "I/O" begin
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


# this method is called to determine whether the callback should be activated
function (solution_callback::SaveSolutionCallback)(u, t, integrator)
  @unpack interval, save_final_solution = solution_callback

  return interval > 0 && (
    (integrator.iter % interval == 0) || (save_final_solution && isfinished(integrator)))
end


# this method is called when the callback is activated
function (solution_callback::SaveSolutionCallback)(integrator)
  u_ode = integrator.u
  @unpack t, dt, iter = integrator
  semi = integrator.p
  mesh, _, _, _ = mesh_equations_solver_cache(semi)

  @trixi_timeit timer() "I/O" begin
    @trixi_timeit timer() "save mesh" if mesh.unsaved_changes
      mesh.current_filename = save_mesh_file(mesh, solution_callback.output_directory, iter)
      mesh.unsaved_changes = false
    end

    element_variables = Dict{Symbol, Any}()
    @trixi_timeit timer() "get element variables" begin
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
    end

    @trixi_timeit timer() "save solution" save_solution_file(u_ode, t, dt, iter, semi, solution_callback, element_variables)
  end

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)
  return nothing
end


@inline function save_solution_file(u_ode, t, dt, iter,
                                    semi::AbstractSemidiscretization, solution_callback,
                                    element_variables=Dict{Symbol,Any}())
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  u = wrap_array_native(u_ode, mesh, equations, solver, cache)
  save_solution_file(u, t, dt, iter, mesh, equations, solver, cache, solution_callback, element_variables)
end


# TODO: Taal refactor, move save_mesh_file?
# function save_mesh_file(mesh::TreeMesh, output_directory, timestep=-1) in io/io.jl

include_fast("save_solution_dg.jl")
