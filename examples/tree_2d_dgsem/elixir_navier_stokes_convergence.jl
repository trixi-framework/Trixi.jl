using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

reynolds_number() = 100
prandtl_number() = 0.72

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesEquations2D(equations, Reynolds=reynolds_number(), Prandtl=prandtl_number(),
                                                          Mach_freestream=0.5)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                periodicity=(true, false),
                n_cells_max=30_000) # set maximum capacity of tree data structure

initial_condition = initial_condition_navier_stokes_convergence_test

# BC types
velocity_bc_top_bottom = NoSlip((x, t, equations) -> initial_condition_navier_stokes_convergence_test(x, t, equations)[2:3])
heat_bc_top_bottom = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_top_bottom = BoundaryConditionViscousWall(velocity_bc_top_bottom, heat_bc_top_bottom)

# define inviscid boundary conditions
boundary_conditions = (; x_neg = boundary_condition_periodic,
                         x_pos = boundary_condition_periodic,
                         y_neg = boundary_condition_slip_wall,
                         y_pos = boundary_condition_slip_wall)

# define viscous boundary conditions
boundary_conditions_parabolic = (; x_neg = boundary_condition_periodic,
                                   x_pos = boundary_condition_periodic,
                                   y_neg = boundary_condition_top_bottom,
                                   y_pos = boundary_condition_top_bottom)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver;
                                             boundary_conditions=(boundary_conditions, boundary_conditions_parabolic),
                                             source_terms=source_terms_navier_stokes_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
callbacks = CallbackSet(summary_callback, alive_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(), abstol=time_int_tol, reltol=time_int_tol, dt = 1e-5,
            save_everystep=false, callback=callbacks)
summary_callback() # print the timer summary

