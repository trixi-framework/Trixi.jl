# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

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

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SaveRestartCallback})
    @nospecialize cb # reduce precompilation time

    restart_callback = cb.affect!
    print(io, "SaveRestartCallback(interval=", restart_callback.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SaveRestartCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        save_restart_callback = cb.affect!

        setup = [
            "interval" => save_restart_callback.interval,
            "save final solution" => save_restart_callback.save_final_restart ? "yes" :
                                     "no",
            "output directory" => abspath(normpath(save_restart_callback.output_directory)),
        ]
        summary_box(io, "SaveRestartCallback", setup)
    end
end

function SaveRestartCallback(; interval = 0,
                             save_final_restart = true,
                             output_directory = "out")
    restart_callback = SaveRestartCallback(interval, save_final_restart,
                                           output_directory)

    DiscreteCallback(restart_callback, restart_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: SaveRestartCallback}
    restart_callback = cb.affect!

    mpi_isroot() && mkpath(restart_callback.output_directory)

    semi = integrator.p
    mesh, _, _, _ = mesh_equations_solver_cache(semi)
    @trixi_timeit timer() "I/O" begin
        if mesh.unsaved_changes
            mesh.current_filename = save_mesh_file(mesh,
                                                   restart_callback.output_directory)
            mesh.unsaved_changes = false
        end
    end

    return nothing
end

# this method is called to determine whether the callback should be activated
function (restart_callback::SaveRestartCallback)(u, t, integrator)
    @unpack interval, save_final_restart = restart_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval > 0 && (((integrator.stats.naccept % interval == 0) &&
             !(integrator.stats.naccept == 0 && integrator.iter > 0)) ||
            (save_final_restart && isfinished(integrator)))
end

# this method is called when the callback is activated
function (restart_callback::SaveRestartCallback)(integrator)
    u_ode = integrator.u
    @unpack t, dt = integrator
    iter = integrator.stats.naccept
    semi = integrator.p
    mesh, _, _, _ = mesh_equations_solver_cache(semi)

    @trixi_timeit timer() "I/O" begin
        if mesh.unsaved_changes
            mesh.current_filename = save_mesh_file(mesh,
                                                   restart_callback.output_directory,
                                                   iter)
            mesh.unsaved_changes = false
        end

        save_restart_file(u_ode, t, dt, iter, semi, restart_callback)
        if integrator.opts.adaptive
            save_restart_controller(mesh, integrator, integrator.opts.controller,
                                    restart_callback)
        end
    end

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end

@inline function save_restart_file(u_ode, t, dt, iter,
                                   semi::AbstractSemidiscretization, restart_callback)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array_native(u_ode, mesh, equations, solver, cache)
    save_restart_file(u, t, dt, iter, mesh, equations, solver, cache, restart_callback)
end

"""
    load_time(restart_file::AbstractString)

Load the time saved in a `restart_file`.
"""
function load_time(restart_file::AbstractString)
    h5open(restart_file, "r") do file
        read(attributes(file)["time"])
    end
end

"""
    load_timestep(restart_file::AbstractString)

Load the time step number (`iter` in OrdinaryDiffEq.jl) saved in a `restart_file`.
"""
function load_timestep(restart_file::AbstractString)
    h5open(restart_file, "r") do file
        read(attributes(file)["timestep"])
    end
end

"""
    load_dt(restart_file::AbstractString)

Load the time step size (`dt` in OrdinaryDiffEq.jl) saved in a `restart_file`.
"""
function load_dt(restart_file::AbstractString)
    h5open(restart_file, "r") do file
        read(attributes(file)["dt"])
    end
end

function load_restart_file(semi::AbstractSemidiscretization, restart_file)
    load_restart_file(mesh_equations_solver_cache(semi)..., restart_file)
end

function load_controller!(integrator, restart_file::AbstractString)
    controller_type = ""
    h5open(restart_file, "r") do file
        if "controller_type" in keys(attributes(file))
            controller_type = read(attributes(file)["controller_type"])
        end
    end
    if controller_type == "PID"
        load_PIDController!(integrator, restart_file)
    elseif controller_type == "PI" || controller_type == "I"
        load_PI_I_Controller!(integrator, restart_file)
    end
    integrator.accept_step = true
end

function load_PIDController!(integrator, restart_file::AbstractString)
    controller = integrator.opts.controller
    if mpi_isroot()
        h5open(restart_file, "r") do file
            integrator.qold = read(attributes(file)["qold"])
            integrator.dtpropose = read(attributes(file)["dtpropose"])
            err = read(file["controller_err"])
            controller.err[1] = err[1]
            controller.err[2] = err[2]
            controller.err[3] = err[3]
        end
    end
    if mpi_isparallel()
        recv_buf = [integrator.qold, integrator.dtpropose]
        append!(recv_buf, controller.err)
        MPI.Bcast!(recv_buf, mpi_root(), mpi_comm())
        integrator.qold = recv_buf[1]
        integrator.dtpropose = recv_buf[2]
        controller.err[1] = recv_buf[3]
        controller.err[2] = recv_buf[4]
        controller.err[3] = recv_buf[5]
    end
end
function load_PI_I_Controller!(integrator, restart_file::AbstractString)
    controller = integrator.opts.controller
    if mpi_isroot()
        h5open(restart_file, "r") do file
            integrator.qold = read(attributes(file)["qold"])
            integrator.dtpropose = read(attributes(file)["dtpropose"])
        end
    end
    if mpi_isparallel()
        recv_buf = [integrator.qold, integrator.dtpropose]
        MPI.Bcast!(recv_buf, mpi_root(), mpi_comm())
        integrator.qold, integrator.dtpropose = recv_buf
    end
end

include("save_restart_dg.jl")
end # @muladd
