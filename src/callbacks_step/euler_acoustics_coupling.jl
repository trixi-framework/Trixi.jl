# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    EulerAcousticsCouplingCallback

!!! warning "Experimental code"
    This callback is experimental and may change in any future release.

A callback that couples the acoustic perturbation equations and compressible Euler equations. Must
be used in conjunction with [`SemidiscretizationEulerAcoustics`](@ref).
This callback manages the flow solver - which is always one time step ahead of the
acoustics solver - and calculates the acoustic source term after each time step. The linearized
Lamb vector is used as the source term, i.e.
```math
\mathbf{s} = -(\mathbf{\omega'} \times \bar{\mathbf{v}}
  + \bar{\mathbf{\omega}} \times \mathbf{v'}),
```
where ``\mathbf{v}`` denotes the velocity, ``\mathbf{\omega}`` denotes the vorticity, the bar
``\bar{(\cdot)}`` indicates time-averaged quantities (see [`AveragingCallback`](@ref)) and prime
``(\cdot)'`` denotes perturbed quantities defined by ``\phi' = \phi - \bar{\phi}``. Note that
the perturbed quantities here are based entirely on the pure flow solution and should not be
confused with the state variables of the acoustic perturbation equations.

In addition, this callback manages the time step size for both solvers
and initializes the mean values of the acoustic perturbation equations using results obtained with
the [`AveragingCallback`](@ref).

- Michael Schlottke-Lakemper (2017)
  A direct-hybrid method for aeroacoustic analysis
  [DOI: 10.18154/RWTH-2017-04082](https://doi.org/10.18154/RWTH-2017-04082)
"""
mutable struct EulerAcousticsCouplingCallback{RealT <: Real, MeanValues,
                                              IntegratorEuler}
    stepsize_callback_acoustics::StepsizeCallback{RealT}
    stepsize_callback_euler::StepsizeCallback{RealT}
    mean_values::MeanValues
    integrator_euler::IntegratorEuler
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any, <:EulerAcousticsCouplingCallback})
    @nospecialize cb # reduce precompilation time
    euler_acoustics_coupling = cb.affect!

    print(io, "EulerAcousticsCouplingCallback(")
    print(io, euler_acoustics_coupling.stepsize_callback_acoustics)
    print(io, ", ", euler_acoustics_coupling.stepsize_callback_euler, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:EulerAcousticsCouplingCallback})
    @nospecialize cb # reduce precompilation time
    euler_acoustics_coupling = cb.affect!

    summary_header(io, "EulerAcousticsCouplingCallback")
    summary_line(io, "acoustics StepsizeCallback",
                 euler_acoustics_coupling.stepsize_callback_acoustics)
    summary_line(io, "Euler StepsizeCallback",
                 euler_acoustics_coupling.stepsize_callback_euler)
    summary_footer(io)
end

"""
    EulerAcousticsCouplingCallback(ode_euler,
                                   averaging_callback::DiscreteCallback{<:Any, <:AveragingCallback},
                                   alg, cfl_acoustics::Real, cfl_euler::Real; kwargs...)

!!! warning "Experimental code"
    This callback is experimental and may change in any future release.

Creates an [`EulerAcousticsCouplingCallback`](@ref) based on the pure flow `ODEProblem` given by
`ode_euler`. Creates an integrator using the time integration method `alg` and the keyword arguments
to solve `ode_euler` (consult the [OrdinaryDiffEq documentation](https://diffeq.sciml.ai/stable/)
for further information).
Manages the step size for both solvers by using the minimum of the maximum step size obtained with
CFL numbers `cfl_acoustics` for the acoustics solver and `cfl_euler` for and flow solver,
respectively.
The mean values for the acoustic perturbation equations are read from `averaging_callback`
(see [`AveragingCallback`](@ref)).
"""
function EulerAcousticsCouplingCallback(ode_euler,
                                        averaging_callback::DiscreteCallback{<:Any,
                                                                             <:AveragingCallback},
                                        alg, cfl_acoustics::Real, cfl_euler::Real;
                                        kwargs...)
    @unpack mean_values = averaging_callback.affect!

    return EulerAcousticsCouplingCallback(ode_euler, mean_values, alg, cfl_acoustics,
                                          cfl_euler;
                                          kwargs...)
end

"""
    EulerAcousticsCouplingCallback(ode_euler, averaging_file::AbstractString, alg,
                                   cfl_acoustics::Real, cfl_euler::Real; kwargs...)

!!! warning "Experimental code"
    This callback is experimental and may change in any future release.

Creates an [`EulerAcousticsCouplingCallback`](@ref) based on the pure flow `ODEProblem` given by
`ode_euler`. Creates an integrator using the time integration method `alg` and the keyword arguments
to solve `ode_euler` (consult the [OrdinaryDiffEq documentation](https://diffeq.sciml.ai/stable/)
for further information).
Manages the step size for both solvers by using the minimum of the maximum step size obtained with
CFL numbers `cfl_acoustics` for the acoustics solver and `cfl_euler` for and flow solver,
respectively.
The mean values for the acoustic perturbation equations are read from `averaging_file`
(see [`AveragingCallback`](@ref)).
"""
function EulerAcousticsCouplingCallback(ode_euler, averaging_file::AbstractString, alg,
                                        cfl_acoustics::Real, cfl_euler::Real; kwargs...)
    semi_euler = ode_euler.p
    mean_values = load_averaging_file(averaging_file, semi_euler)

    return EulerAcousticsCouplingCallback(ode_euler, mean_values, alg, cfl_acoustics,
                                          cfl_euler;
                                          kwargs...)
end

function EulerAcousticsCouplingCallback(ode_euler, mean_values, alg, cfl_acoustics,
                                        cfl_euler;
                                        kwargs...)
    # Set up ODE Integrator for Euler equations
    integrator_euler = init(ode_euler, alg, save_everystep = false, dt = 1.0; kwargs...) # dt will be overwritten

    euler_acoustics_coupling = EulerAcousticsCouplingCallback{typeof(cfl_acoustics),
                                                              typeof(mean_values),
                                                              typeof(integrator_euler)}(StepsizeCallback(cfl_acoustics),
                                                                                        StepsizeCallback(cfl_euler),
                                                                                        mean_values,
                                                                                        integrator_euler)
    condition = (u, t, integrator) -> true

    return DiscreteCallback(condition, euler_acoustics_coupling,
                            save_positions = (false, false),
                            initialize = initialize!)
end

# This is called before the main loop and initializes the mean values in u_ode
function initialize!(cb::DiscreteCallback{Condition, Affect!}, u_ode, t,
                     integrator_acoustics) where {Condition,
                                                  Affect! <:
                                                  EulerAcousticsCouplingCallback}
    euler_acoustics_coupling = cb.affect!
    semi = integrator_acoustics.p
    @unpack semi_acoustics = semi

    # Initialize mean values in u_ode
    u_acoustics = wrap_array(u_ode, semi_acoustics)
    @unpack mean_values = euler_acoustics_coupling
    @views @. u_acoustics[4:5, .., :] = mean_values.v_mean
    @views @. u_acoustics[6, .., :] = mean_values.c_mean
    @views @. u_acoustics[7, .., :] = mean_values.rho_mean

    # Adjust stepsize, advance the flow solver by one time step
    cb.affect!(integrator_acoustics)

    return nothing
end

# This function is called at the end of every time step and advances the Euler solution by one
# time step, manages the time stepsize for both the acoustics and Euler solvers and calculates the
# acoustic sources for the next acoustics time step
function (euler_acoustics_coupling::EulerAcousticsCouplingCallback)(integrator_acoustics)
    @unpack stepsize_callback_acoustics, stepsize_callback_euler, integrator_euler = euler_acoustics_coupling

    @assert integrator_acoustics.t == integrator_euler.t

    # Use the minimum of the acoustics and Euler stepsizes for both solvers
    stepsize_callback_acoustics(integrator_acoustics)
    stepsize_callback_euler(integrator_euler)
    dt = min(get_proposed_dt(integrator_acoustics), get_proposed_dt(integrator_euler))

    set_proposed_dt!(integrator_acoustics, dt)
    integrator_acoustics.opts.dtmax = dt
    integrator_acoustics.dtcache = dt

    set_proposed_dt!(integrator_euler, dt)
    integrator_euler.opts.dtmax = dt
    integrator_euler.dtcache = dt

    # Advance the Euler solution by one step and check for errors
    if !isfinished(integrator_euler)
        @trixi_timeit timer() "Euler solver" step!(integrator_euler)
        return_code = check_error(integrator_euler)
        if !(SciMLBase.successful_retcode(return_code) ||
             return_code != SciMLBase.ReturnCode.Default)
            error("Error during compressible Euler time integration. Received return code $(return_code)")
        end
    end

    # Calculate acoustic sources based on linearized lamb vector
    semi = integrator_acoustics.p
    semi_euler = integrator_euler.p
    u_acoustics = wrap_array(integrator_acoustics.u, semi)
    u_euler = wrap_array(integrator_euler.u, semi_euler)
    @unpack acoustic_source_terms, coupled_element_ids = semi.cache
    @unpack vorticity_mean = euler_acoustics_coupling.mean_values

    @trixi_timeit timer() "calc acoustic source terms" begin
        calc_acoustic_sources!(acoustic_source_terms, u_euler, u_acoustics,
                               vorticity_mean, coupled_element_ids,
                               mesh_equations_solver_cache(semi_euler)...)
    end

    # avoid re-evaluation possible FSAL stages
    u_modified!(integrator_acoustics, false)
    u_modified!(integrator_euler, false)

    return nothing
end

include("euler_acoustics_coupling_dg2d.jl")
end # @muladd
