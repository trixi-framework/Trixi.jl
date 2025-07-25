using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the acoustic perturbation equations

equations = AcousticPerturbationEquations2D(v_mean_global = (0.0, -0.5),
                                            c_mean_global = 1.0,
                                            rho_mean_global = 1.0)

# Create DG solver with polynomial degree = 4 and (local) Lax-Friedrichs/Rusanov flux

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
solver = DGSEM(polydeg = 4, surface_flux = FluxLaxFriedrichs(max_abs_speed_naive))

# Create unstructured quadrilateral mesh from a file
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/3c79baad6b4d73bb26ec6420b5d16f45/raw/22aefc4ec2107cf0bffc40e81dfbc52240c625b1/mesh_five_circles_in_circle.mesh",
                           joinpath(@__DIR__, "mesh_five_circles_in_circle.mesh"))

mesh = UnstructuredMesh2D(mesh_file)

"""
    initial_condition_gauss_wall(x, t, equations::AcousticPerturbationEquations2D)

A Gaussian pulse, used in the `gauss_wall` example elixir in combination with
[`boundary_condition_wall`](@ref). Uses the global mean values from `equations`.
"""
function initial_condition_gauss_wall(x, t, equations::AcousticPerturbationEquations2D)
    v1_prime = 0.0
    v2_prime = 0.0
    p_prime = exp(-log(2) * (x[1]^2 + (x[2] - 25)^2) / 25)

    prim = SVector(v1_prime, v2_prime, p_prime, global_mean_vars(equations)...)

    return prim2cons(prim, equations)
end
initial_condition = initial_condition_gauss_wall

boundary_conditions = Dict(:OuterCircle => boundary_condition_slip_wall,
                           :InnerCircle1 => boundary_condition_slip_wall,
                           :InnerCircle2 => boundary_condition_slip_wall,
                           :InnerCircle3 => boundary_condition_slip_wall,
                           :InnerCircle4 => boundary_condition_slip_wall,
                           :InnerCircle5 => boundary_condition_slip_wall)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 300.0
tspan = (0.0, 300.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 50, solution_variables = cons2state)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-6, reltol = 1.0e-6,
            ode_default_options()..., callback = callbacks);
