
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

  mpi_isroot() && mkpath(restart_callback.output_directory)

  semi = integrator.p
  mesh, _, _, _ = mesh_equations_solver_cache(semi)
  @timeit_debug timer() "I/O" begin
    if mesh.unsaved_changes
      mesh.current_filename = save_mesh_file(mesh, restart_callback.output_directory)
      mesh.unsaved_changes = false
    end
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


"""
    load_mesh(restart_file::AbstractString; n_cells_max)

Load the mesh from the `restart_file`.
"""
function load_mesh(restart_file::AbstractString; n_cells_max)
  load_mesh(restart_file, mpi_parallel(); n_cells_max=n_cells_max)
end

function load_mesh(restart_file::AbstractString, mpi_parallel::Val{false};
                   n_cells_max)
  ndims_ = h5open(restart_file, "r") do file
    read(attrs(file)["ndims"])
  end

  mesh = TreeMesh(SerialTree{ndims_}, n_cells_max)
  load_mesh!(mesh, restart_file)
end

function load_mesh!(mesh::SerialTreeMesh, restart_file::AbstractString)
  # Determine mesh filename
  filename = get_restart_mesh_filename(restart_file, mpi_parallel(mesh))
  mesh.current_filename = filename
  mesh.unsaved_changes = false

  # Read mesh file
  h5open(filename, "r") do file
    # Set domain information
    mesh.tree.center_level_0 = read(attrs(file)["center_level_0"])
    mesh.tree.length_level_0 = read(attrs(file)["length_level_0"])
    mesh.tree.periodicity    = Tuple(read(attrs(file)["periodicity"]))

    # Set length
    n_cells = read(attrs(file)["n_cells"])
    resize!(mesh.tree, n_cells)

    # Read in data
    mesh.tree.parent_ids[1:n_cells] = read(file["parent_ids"])
    mesh.tree.child_ids[:, 1:n_cells] = read(file["child_ids"])
    mesh.tree.neighbor_ids[:, 1:n_cells] = read(file["neighbor_ids"])
    mesh.tree.levels[1:n_cells] = read(file["levels"])
    mesh.tree.coordinates[:, 1:n_cells] = read(file["coordinates"])
  end

  return mesh
end


function load_mesh(restart_file::AbstractString, mpi_parallel::Val{true};
                   n_cells_max)
  ndims_ = h5open(restart_file, "r") do file
    read(attrs(file)["ndims"])
  end

  mesh = TreeMesh(ParallelTree{ndims_}, n_cells_max)
  load_mesh!(mesh, restart_file)
end

function load_mesh!(mesh::ParallelTreeMesh, restart_file::AbstractString)
  # Determine mesh filename
  filename = get_restart_mesh_filename(restart_file, mpi_parallel(mesh))
  mesh.current_filename = filename
  mesh.unsaved_changes = false

  if mpi_isroot()
    h5open(filename, "r") do file
      # Set domain information
      mesh.tree.center_level_0 = read(attrs(file)["center_level_0"])
      mesh.tree.length_level_0 = read(attrs(file)["length_level_0"])
      mesh.tree.periodicity    = Tuple(read(attrs(file)["periodicity"]))
      MPI.Bcast!(collect(mesh.tree.center_level_0), mpi_root(), mpi_comm())
      MPI.Bcast!(collect(mesh.tree.length_level_0), mpi_root(), mpi_comm())
      MPI.Bcast!(collect(mesh.tree.periodicity),    mpi_root(), mpi_comm())

      # Set length
      n_cells = read(attrs(file)["n_cells"])
      MPI.Bcast!(Ref(n_cells), mpi_root(), mpi_comm())
      resize!(mesh.tree, n_cells)

      # Read in data
      mesh.tree.parent_ids[1:n_cells] = read(file["parent_ids"])
      mesh.tree.child_ids[:, 1:n_cells] = read(file["child_ids"])
      mesh.tree.neighbor_ids[:, 1:n_cells] = read(file["neighbor_ids"])
      mesh.tree.levels[1:n_cells] = read(file["levels"])
      mesh.tree.coordinates[:, 1:n_cells] = read(file["coordinates"])
      @views MPI.Bcast!(mesh.tree.parent_ids[1:n_cells],      mpi_root(), mpi_comm())
      @views MPI.Bcast!(mesh.tree.child_ids[:, 1:n_cells],    mpi_root(), mpi_comm())
      @views MPI.Bcast!(mesh.tree.neighbor_ids[:, 1:n_cells], mpi_root(), mpi_comm())
      @views MPI.Bcast!(mesh.tree.levels[1:n_cells],          mpi_root(), mpi_comm())
      @views MPI.Bcast!(mesh.tree.coordinates[:, 1:n_cells],  mpi_root(), mpi_comm())
    end
  else # non-root ranks
    # Set domain information
    mesh.tree.center_level_0 = MPI.Bcast!(collect(mesh.tree.center_level_0), mpi_root(), mpi_comm())
    mesh.tree.length_level_0 = MPI.Bcast!(collect(mesh.tree.length_level_0), mpi_root(), mpi_comm())[1]
    mesh.tree.periodicity    = Tuple(MPI.Bcast!(collect(mesh.tree.periodicity),    mpi_root(), mpi_comm()))

    # Set length
    n_cells = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())[]
    resize!(mesh.tree, n_cells)

    # Read in data
    @views MPI.Bcast!(mesh.tree.parent_ids[1:n_cells],      mpi_root(), mpi_comm())
    @views MPI.Bcast!(mesh.tree.child_ids[:, 1:n_cells],    mpi_root(), mpi_comm())
    @views MPI.Bcast!(mesh.tree.neighbor_ids[:, 1:n_cells], mpi_root(), mpi_comm())
    @views MPI.Bcast!(mesh.tree.levels[1:n_cells],          mpi_root(), mpi_comm())
    @views MPI.Bcast!(mesh.tree.coordinates[:, 1:n_cells],  mpi_root(), mpi_comm())
  end

  # Partition mesh
  partition!(mesh)

  return mesh
end


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
