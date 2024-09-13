# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    AveragingCallback(semi::SemidiscretizationHyperbolic, tspan; output_directory="out",
                      filename="averaging.h5")

!!! warning "Experimental code"
    This callback is experimental and may change in any future release.

A callback that averages the flow field described by `semi` which must be a semidiscretization of
the compressible Euler equations in two dimensions. The callback records the mean velocity,
mean speed of sound, mean density, and mean vorticity for each node over the time interval given by
`tspan` and stores the results in an HDF5 file `filename` in the directory `output_directory`. Note
that this callback does not support adaptive mesh refinement ([`AMRCallback`](@ref)).
"""
struct AveragingCallback{TSpan, MeanValues, Cache}
    tspan::TSpan
    mean_values::MeanValues
    cache::Cache
    output_directory::String
    filename::String
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:AveragingCallback})
    @nospecialize cb # reduce precompilation time
    averaging_callback = cb.affect!
    @unpack tspan = averaging_callback

    print(io, "AveragingCallback(tspan=", tspan, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:AveragingCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        averaging_callback = cb.affect!

        setup = [
            "Start time" => first(averaging_callback.tspan),
            "Final time" => last(averaging_callback.tspan)
        ]
        summary_box(io, "AveragingCallback", setup)
    end
end

function AveragingCallback(semi::SemidiscretizationHyperbolic{<:Any,
                                                              <:CompressibleEulerEquations2D},
                           tspan; output_directory = "out", filename = "averaging.h5")
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    mean_values = initialize_mean_values(mesh, equations, solver, cache)
    cache = create_cache(AveragingCallback, mesh, equations, solver, cache)

    averaging_callback = AveragingCallback(tspan, mean_values, cache, output_directory,
                                           filename)
    condition = (u, t, integrator) -> first(tspan) <= t <= last(tspan)

    return DiscreteCallback(condition, averaging_callback,
                            save_positions = (false, false),
                            initialize = initialize!)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u_ode, t,
                     integrator) where {Condition, Affect! <: AveragingCallback}
    averaging_callback = cb.affect!
    semi = integrator.p
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    @trixi_timeit timer() "averaging" initialize_cache!(averaging_callback.cache, u,
                                                        mesh, equations, solver, cache)

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end

# This function is called during time integration and updates the mean values according to the
# trapezoidal rule
function (averaging_callback::AveragingCallback)(integrator)
    @unpack mean_values = averaging_callback

    u_ode = integrator.u
    u_prev_ode = integrator.uprev
    semi = integrator.p
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)
    u_prev = wrap_array(u_prev_ode, mesh, equations, solver, cache)

    dt = integrator.t - integrator.tprev
    tspan = averaging_callback.tspan

    integration_constant = 0.5 * dt / (tspan[2] - tspan[1]) # .5 due to trapezoidal rule

    @trixi_timeit timer() "averaging" calc_mean_values!(mean_values,
                                                        averaging_callback.cache,
                                                        u, u_prev, integration_constant,
                                                        mesh, equations, solver, cache)

    # Store mean values in a file if this is the last time step
    if isfinished(integrator)
        save_averaging_file(averaging_callback, semi)
    end

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)

    return nothing
end

function save_averaging_file(averaging_callback, semi::AbstractSemidiscretization)
    # Create output directory if it doesn't exist
    mkpath(averaging_callback.output_directory)

    save_averaging_file(averaging_callback, mesh_equations_solver_cache(semi)...)
end

function load_averaging_file(averaging_file, semi::AbstractSemidiscretization)
    load_averaging_file(averaging_file, mesh_equations_solver_cache(semi)...)
end

include("averaging_dg.jl")
include("averaging_dg2d.jl")
end # @muladd
