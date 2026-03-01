using Trixi
using OrdinaryDiffEqCore: NewIController

###############################################################################
# create a restart file

elixir_file = "elixir_advection_implicit_sparse_jacobian.jl"
restart_file = "restart_000000006.h5"

trixi_include(@__MODULE__, joinpath(@__DIR__, elixir_file))

###############################################################################

restart_filename = joinpath("out", restart_file)
tspan = (load_time(restart_filename), 2.0)
dt_restart = load_dt(restart_filename)

ode_jac_sparse = semidiscretize(semi_float_type, tspan,
                                restart_filename,
                                jac_prototype = jac_prototype,
                                colorvec = coloring_vec)

###############################################################################
# run the simulation

# Use an IController to ensure reproducible restart behavior.
# Unlike PIController, IController has no memory of previous error estimates,
# so restarting from a checkpoint produces deterministic step size selection.
# Setting qmax_first_step = qmax avoids first-step acceleration that would
# cause the restarted solve to diverge from a continuous run.
# See https://github.com/SciML/OrdinaryDiffEq.jl/issues/3101
alg = TRBDF2(; autodiff = AutoFiniteDiff())
controller = NewIController(alg, qmax = 10, qmax_first_step = 10)

sol = solve(ode_jac_sparse, alg;
            adaptive = true, dt = dt_restart, save_everystep = false,
            callback = callbacks, controller = controller);
