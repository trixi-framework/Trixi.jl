# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    GlmSpeedCallback(; glm_scale=0.5, cfl, semi_indices=())

Update the divergence cleaning wave speed `c_h` according to the time step
computed in [`StepsizeCallback`](@ref) for the ideal GLM-MHD equations.
The `cfl` number should be set to the same value as for the time step size calculation. The
`glm_scale` ensures that the GLM wave speed is lower than the fastest physical waves in the MHD
solution and should thus be set to a value within the interval [0,1]. Note that `glm_scale = 0`
deactivates the divergence cleaning.

In case of coupled semidiscretizations, specify for which `semi_index`, i.e. index of the
semidiscretization, the divergence cleaning should be applied. See also
[`SemidiscretizationCoupled`](@ref).
Note: `SemidiscretizationCoupled` and all related features are considered experimental and
may change at any time.
"""
struct GlmSpeedCallback{RealT <: Real}
    glm_scale::RealT
    cfl::RealT
    semi_indices::Vector{Int}
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:GlmSpeedCallback})
    @nospecialize cb # reduce precompilation time

    glm_speed_callback = cb.affect!
    @unpack glm_scale, cfl, semi_indices = glm_speed_callback
    print(io, "GlmSpeedCallback(glm_scale=", glm_scale, ", cfl=", cfl, "semi_indices=",
          semi_indices, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:GlmSpeedCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        glm_speed_callback = cb.affect!

        setup = [
            "GLM wave speed scaling" => glm_speed_callback.glm_scale,
            "Expected CFL number" => glm_speed_callback.cfl,
            "Selected semidiscretizations" => glm_speed_callback.semi_indices
        ]
        summary_box(io, "GlmSpeedCallback", setup)
    end
end

function GlmSpeedCallback(; glm_scale = 0.5, cfl, semi_indices = Int[])
    @assert 0<=glm_scale<=1 "glm_scale must be between 0 and 1"

    glm_speed_callback = GlmSpeedCallback(glm_scale, cfl, semi_indices)

    DiscreteCallback(glm_speed_callback, glm_speed_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: GlmSpeedCallback}
    cb.affect!(integrator)
end

# this method is called to determine whether the callback should be activated
function (glm_speed_callback::GlmSpeedCallback)(u, t, integrator)
    return true
end

function update_cleaning_speed!(semi, glm_speed_callback, dt)
    @unpack glm_scale, cfl = glm_speed_callback

    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

    # compute time step for GLM linear advection equation with c_h=1 (redone due to the possible AMR)
    c_h_deltat = calc_dt_for_cleaning_speed(cfl, mesh, equations, solver, cache)

    # c_h is proportional to its own time step divided by the complete MHD time step
    # We use @reset here since the equations are immutable (to work on GPUs etc.).
    # Thus, we need to modify the equations field of the semidiscretization.
    @reset equations.c_h = glm_scale * c_h_deltat / dt
    semi.equations = equations

    return semi
end

# This method is called as callback after the StepsizeCallback during the time integration.
@inline function (glm_speed_callback::GlmSpeedCallback)(integrator)
    dt = get_proposed_dt(integrator)
    semi = integrator.p

    # Call the appropriate update function (this indirection allows to specialize on,
    # e.g., the semidiscretization type)
    update_cleaning_speed!(semi, glm_speed_callback, dt)

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)

    return nothing
end

include("glm_speed_dg.jl")
end # @muladd
