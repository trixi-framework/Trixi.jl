
"""
    SaveRestartCallback(; interval=0,
                          save_final_restart=true,
                          output_directory="out")

Save the current numerical solution in a restart file every `interval` time steps.
"""
mutable struct SaveRestartCallback
  interval::Int
  save_final_restart::Bool
  output_directory::String
end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:SaveRestartCallback}
  restart_callback = cb.affect!
  print(io, "SaveRestartCallback(interval=", restart_callback.interval, ")")
end
# TODO: Taal bikeshedding, implement a method with more information and the signature
# function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:SaveRestartCallback}
# end


function SaveRestartCallback(; interval=0,
                               save_final_restart=true,
                               output_directory="out")
  # Checking for floating point equality is OK here as `DifferentialEquations.jl`
  # sets the time exactly to the final time in the last iteration
  condition = (u, t, integrator) -> interval > 0 && ((integrator.iter % interval == 0) ||
                                                     (save_final_restart && (t == integrator.sol.prob.tspan[2] ||
                                                                              isempty(integrator.opts.tstops))))

  restart_callback = SaveRestartCallback(interval, save_final_restart,
                                         output_directory)

  DiscreteCallback(condition, restart_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:SaveRestartCallback}
  restart_callback = cb.affect!

  mkpath(restart_callback.output_directory)

  semi = integrator.p
  mesh, _, _, _ = mesh_equations_solver_cache(semi)
  @timeit_debug timer() "I/O" begin
    mesh.current_filename = save_mesh_file(mesh, restart_callback.output_directory)
    mesh.unsaved_changes = false
  end

  return nothing
end


function (restart_callback::SaveRestartCallback)(integrator)
  u_ode = integrator.u
  @unpack t, dt, iter = integrator
  semi = integrator.p
  mesh, _, _, _ = mesh_equations_solver_cache(semi)

  @timeit_debug timer() "I/O" begin
    if mesh.unsaved_changes
      mesh.current_filename = save_mesh_file(mesh, restart_callback.output_directory, iter)
      mesh.unsaved_changes = false
    end

    save_restart_file(u_ode, t, dt, iter, semi, restart_callback)
  end

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)
  return nothing
end


@inline function save_restart_file(u_ode::AbstractVector, t, dt, iter,
                                   semi::AbstractSemidiscretization, restart_callback)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  u = wrap_array(u_ode, mesh, equations, solver, cache)
  save_restart_file(u, t, dt, iter, mesh, equations, solver, cache, restart_callback)
end


# TODO: Taal refactor, move save_mesh_file?
# function save_mesh_file(mesh::TreeMesh, output_directory, timestep=-1) in io/io.jl


"""
    load_time(restart_file::AbstractString)

Load the time saved in a `restart_file`.
"""
function load_time(restart_file::AbstractString)
  h5open(restart_file, "r") do file
    read(attrs(file)["time"])
  end
end


function load_restart_file(semi::AbstractSemidiscretization, restart_file)
  load_restart_file(mesh_equations_solver_cache(semi)..., restart_file)
end


include("save_restart_dg.jl")
