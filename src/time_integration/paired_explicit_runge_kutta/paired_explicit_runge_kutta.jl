# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Basic implementation of the second-order paired explicit Runge-Kutta (PERK) method
include("methods_PERK2.jl")
include("methods_PERK3.jl")
# Define all of the functions necessary for polynomial optimizations
include("polynomial_optimizer.jl")

"""
    add_tstop!(integrator::AbstractPairedExplicitRKSingleIntegrator, t)
Add a time stop during the time integration process.
This function is called after the periodic SaveSolutionCallback to specify the next stop to save the solution.
"""
function add_tstop!(integrator::AbstractPairedExplicitRKSingleIntegrator, t)
    integrator.tdir * (t - integrator.t) < zero(integrator.t) &&
        error("Tried to add a tstop that is behind the current time. This is strictly forbidden")
    # We need to remove the first entry of tstops when a new entry is added.
    # Otherwise, the simulation gets stuck at the previous tstop and dt is adjusted to zero.
    if length(integrator.opts.tstops) > 1
        pop!(integrator.opts.tstops)
    end
    push!(integrator.opts.tstops, integrator.tdir * t)
end

has_tstop(integrator::AbstractPairedExplicitRKSingleIntegrator) = !isempty(integrator.opts.tstops)
first_tstop(integrator::AbstractPairedExplicitRKSingleIntegrator) = first(integrator.opts.tstops)

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::AbstractPairedExplicitRKSingle;
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve!(integrator)
end

function solve!(integrator::PairedExplicitRK)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        step!(integrator)
    end # "main loop" timer

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PairedExplicitRK, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)
end

# Add definitions of functions related to polynomial optimization by NLsolve here
# such that hey can be exported from Trixi.jl and extended in the TrixiConvexECOSExt package
# extension or by the NLsolve-specific code loaded by Requires.jl
function solve_a_butcher_coeffs_unknown! end
end # @muladd
