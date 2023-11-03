
using OrdinaryDiffEq
using Trixi

# define new structs inside a module to allow re-evaluating the file
module TrixiExtensionExample

using Trixi
using OrdinaryDiffEq: DiscreteCallback, u_modified!

# This is an example implementation for a simple stage callback (i.e., a callable
# that is executed after each Runge-Kutta *stage*), which records some values
# each time it is called. Its sole purpose here is to showcase how to implement
# a stage callback for Trixi.jl.
struct ExampleStageCallback
    times::Vector{Float64}
    min_values::Vector{Float64}
    max_values::Vector{Float64}

    # You can optionally define an inner constructor like the one below to set up
    # some required stuff. You can also create outer constructors (not demonstrated
    # here) for further customization options.
    function ExampleStageCallback()
        new(Float64[], Float64[], Float64[])
    end
end

# This method is called when the `ExampleStageCallback` is used as `stage_limiter!`
# which gets called after every RK stage. There is no specific initialization
# method for such `stage_limiter!`s in OrdinaryDiffEq.jl.
function (example_stage_callback::ExampleStageCallback)(u_ode, _, semi, t)
    min_val, max_val = extrema(u_ode)
    push!(example_stage_callback.times, t)
    push!(example_stage_callback.min_values, min_val)
    push!(example_stage_callback.max_values, max_val)

    return nothing
end

# This is an example implementation for a simple step callback (i.e., a callable
# that is potentially executed after each Runge-Kutta *step*), which records
# some values each time it is called. Its sole purpose here is to showcase
# how to implement a step callback for Trixi.jl.
struct ExampleStepCallback
    message::String
    times::Vector{Float64}
    min_values::Vector{Float64}
    max_values::Vector{Float64}

    # You can optionally define an inner constructor like the one below to set up
    # some required stuff. You can also create outer constructors (not demonstrated
    # here) for further customization options.
    function ExampleStepCallback(message::String)
        new(message, Float64[], Float64[], Float64[])
    end
end

# This method is called when the `ExampleStepCallback` is used as callback
# which gets called after RK steps.
function (example_callback::ExampleStepCallback)(integrator)
    u_ode = integrator.u
    t = integrator.t
    # You can also access semi = integrator.p

    min_val, max_val = extrema(u_ode)
    push!(example_callback.times, t)
    push!(example_callback.min_values, min_val)
    push!(example_callback.max_values, max_val)

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end

# This method is used to wrap an `ExampleStepCallback` inside a `DiscreteCallback`
# which gets called after every RK step. You can pass an additional initialization
# method and a separate condition specifying whether the callback shall be called.
function ExampleStepCallback(; message::String)
    # Call the `ExampleStepCallback` after every RK step.
    condition = (u_ode, t, integrator) -> true

    # You can optionally pass an initialization method. There, you can access the
    # `ExampleStepCallback` as `cb.affect!`.
    initialize = (cb, u_ode, t, integrator) -> println(cb.affect!.message)

    example_callback = ExampleStepCallback(message)

    DiscreteCallback(condition, example_callback,
                     save_positions = (false, false),
                     initialize = initialize)
end

end # module TrixiExtensionExample

import .TrixiExtensionExample

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy, energy_total))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2cons)

example_callback = TrixiExtensionExample.ExampleStepCallback(message = "안녕하세요?")

stepsize_callback = StepsizeCallback(cfl = 1.6)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        example_callback,
                        stepsize_callback)

# In OrdinaryDiffEq.jl, the `step_limiter!` is called after every Runge-Kutta step
# but before possible RHS evaluations of the new value occur. Hence, it's possible
# to modify the new solution value there without impacting the performance of FSAL
# methods.
# The `stage_limiter!` is called after computing a Runge-Kutta stage value but
# before evaluating the corresponding stage derivatives.
# Hence, if a limiter should be called before each RHS evaluation, it needs to be
# set as `stage_limiter!`.
example_stage_callback! = TrixiExtensionExample.ExampleStageCallback()

###############################################################################
# run the simulation

sol = solve(ode,
            CarpenterKennedy2N54(example_stage_callback!, williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary

# Check whether we recorded the same values.
# Remember that CarpenterKennedy2N54 has five stages per step.
@assert example_stage_callback!.times[5:5:end] ≈ example_callback.affect!.times
@assert example_stage_callback!.min_values[5:5:end] ≈ example_callback.affect!.min_values
@assert example_stage_callback!.max_values[5:5:end] ≈ example_callback.affect!.max_values
