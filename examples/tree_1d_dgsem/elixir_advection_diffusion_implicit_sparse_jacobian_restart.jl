using Trixi

###############################################################################
# create a restart file

elixir_file = "elixir_advection_diffusion_implicit_sparse_jacobian.jl"
restart_file = "restart_000000100.h5"

trixi_include(@__MODULE__, joinpath(@__DIR__, elixir_file))

###############################################################################

restart_filename = joinpath("out", restart_file)
tspan = (load_time(restart_filename), 2.0)
dt_restart = load_dt(restart_filename)

ode = semidiscretize(semi, tspan,
                     restart_filename,
                     jac_prototype_parabolic = jac_prototype_parabolic,
                     colorvec_parabolic = coloring_vec_parabolic)

###############################################################################
# run the simulation

sol = solve(ode, SBDF2(; autodiff = AutoFiniteDiff());
            ode_default_options()...,
            dt = dt_restart,
            abstol = 1e-9, reltol = 1e-9,
            callback = callbacks);
