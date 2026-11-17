using Trixi

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

sol = solve(ode_jac_sparse,
            # Default `AutoForwardDiff()` is not yet working, see
            # https://github.com/trixi-framework/Trixi.jl/issues/2369
            TRBDF2(; autodiff = AutoFiniteDiff());
            adaptive = true, dt = dt_restart, save_everystep = false, callback = callbacks);
