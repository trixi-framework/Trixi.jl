# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

    #! format: noindent

    mutable struct LoadRestartCallback
        restart_file::String
    end

    function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:LoadRestartCallback})
        @nospecialize cb # reduce precompilation time

        load_callback = cb.affect!
        print(io, "LoadRestartCallback(restart file=", load_callback.restart_file, ")")
    end

    function Base.show(io::IO, ::MIME"text/plain",
                       cb::DiscreteCallback{<:Any, <:LoadRestartCallback})
        @nospecialize cb # reduce precompilation time

        if get(io, :compact, false)
            show(io, cb)
        else
            save_load_callback = cb.affect!
            setup = [
                "restart file" => save_load_callback.restart_file]
            summary_box(io, "LoadRestartCallback", setup)
        end
    end

    function LoadRestartCallback(; restart_file = "out.h5")
        load_callback = LoadRestartCallback(restart_file)

        DiscreteCallback(load_callback, load_callback, # the first one is the condition, the second the affect!
                         save_positions = (false, false),
                         initialize = initialize!)
    end

    function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                         integrator) where {Condition, Affect! <: LoadRestartCallback}
        load_callback = cb.affect!
        if integrator.opts.adaptive
            load_integrator!(load_callback.restart_file, integrator,
                             integrator.opts.controller, integrator.alg)
        end
        return nothing
    end

    function load_integrator!(restart_file::String, integrator, controller::PIController,
                              alg)
        if mpi_isroot()
            h5open(restart_file, "r") do file
                integrator.qold = read(attributes(file)["qold"])
                integrator.dt = read(attributes(file)["dtpropose"])
            end
        end
        if mpi_isparallel()
            recv_buf = [integrator.qold, integrator.dt]
            MPI.Bcast!(recv_buf, mpi_root(), mpi_comm())
            integrator.qold, integrator.dt = recv_buf
        end
    end

    function load_integrator!(restart_file::String, integrator, controller::PIDController,
                              alg)
        if mpi_isroot()
            h5open(restart_file, "r") do file
                integrator.qold = read(attributes(file)["qold"])
                integrator.dt = read(attributes(file)["dtpropose"])
                err = read(file["controller_err"])
                controller.err[1] = err[1]
                controller.err[2] = err[2]
                controller.err[3] = err[3]
            end
        end
        if mpi_isparallel()
            recv_buf = [integrator.qold, integrator.dt]
            append!(recv_buf, controller.err)
            MPI.Bcast!(recv_buf, mpi_root(), mpi_comm())
            integrator.qold = recv_buf[1]
            integrator.dt = recv_buf[2]
            controller.err[1] = recv_buf[3]
            controller.err[2] = recv_buf[4]
            controller.err[3] = recv_buf[5]
        end
    end

    function load_integrator!(restart_file::String, integrator, controller, alg)
        h5open(restart_file, "r") do file
            integrator.dt = read(attributes(file)["dtpropose"])
        end
        if mpi_isroot()
            h5open(restart_file, "r") do file
                integrator.dt = read(attributes(file)["dtpropose"])
            end
        end
        if mpi_isparallel()
            recv_buf = [integrator.dt]
            MPI.Bcast!(recv_buf, mpi_root(), mpi_comm())
            integrator.dt = recv_buf[1]
        end
    end

    # this method is called to determine whether the callback should be activated
    (load_callback::LoadRestartCallback)(u, t, integrator) = false

    # this method is called when the callback is activated
    (load_callback::LoadRestartCallback)(integrator) = nothing
end # @muladd
