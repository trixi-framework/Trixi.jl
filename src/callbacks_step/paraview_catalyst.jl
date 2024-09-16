# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct ParaviewCatalystCallback
    interval::Int
end

function Base.show(io::IO,
                   cb::DiscreteCallback{Condition, Affect!}) where {Condition,
                                                                    Affect! <:
                                                                    ParaviewCatalystCallback
                                                                    }
    visualization_callback = cb.affect!
    @unpack interval = visualization_callback
    print(io, "ParaviewCatalystCallback(",
          "interval=", interval,")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{Condition, Affect!}) where {Condition,
                                                                    Affect! <:
                                                                    ParaviewCatalystCallback
                                                                    }
    if get(io, :compact, false)
        show(io, cb)
    else
        visualization_callback = cb.affect!

        setup = [
            "interval" => visualization_callback.interval,
            
        ]
        summary_box(io, "ParaviewCatalystCallback", setup)
    end
end

"""
    ParaviewCatalystCallback(; interval=0,
                            )

Create a callback that visualizes results during a simulation, also known as *in-situ
visualization*.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in any future releases.
"""
function ParaviewCatalystCallback(; interval = 0,
                               )
    mpi_isparallel() && error("this callback does not work in parallel yet")


    visualization_callback = ParaviewCatalystCallback(interval)

    # Warn users if they create a visualization callback without having loaded the Plots package
    #
    # Note: This warning is added for convenience, as Plots is the only "officially" supported
    #       visualization package right now. However, in general nothing prevents anyone from using
    #       other packages such as Makie, Gadfly etc., given that appropriate `plot_creator`s are
    #       passed. This is also the reason why the visualization callback is not included via
    #       Requires.jl only when Plots is present.
    #       In the future, we should update/remove this warning if other plotting packages are
    #       starting to be used.
    if !(:ParaviewCatalyst in names(@__MODULE__, all = true))
        @warn "Package `ParaviewCatalyst` not loaded but required by `ParaviewCatalystCallback` to visualize results"
    end

    DiscreteCallback(visualization_callback, visualization_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: ParaviewCatalystCallback}
    visualization_callback = cb.affect!

    visualization_callback(integrator)

    return nothing
end

# this method is called to determine whether the callback should be activated
function (visualization_callback::ParaviewCatalystCallback)(u, t, integrator)
    @unpack interval = visualization_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval > 0 && (integrator.stats.naccept % interval == 0 ||
            isfinished(integrator))
end

# this method is called when the callback is activated
function (visualization_callback::ParaviewCatalystCallback)(integrator)
    u_ode = integrator.u
    semi = integrator.p
    time = integrator.t
    timestep = integrator.stats.naccept

    println("***Catalyst Callback activated")
    
    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end

end # @muladd
