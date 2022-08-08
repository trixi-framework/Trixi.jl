using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

reynolds_number() = 100
prandtl_number() = 0.72

equations = CompressibleEulerEquations2D(1.4)
# Note: If you change the Navier-Stokes parameters here, also change them in the initial condition
# I really do not like this structure but it should work for now
equations_parabolic = CompressibleNavierStokesEquations2D(equations, Reynolds=reynolds_number(), Prandtl=prandtl_number(),
                                                          Mach_freestream=0.5)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
dg = DGMulti(polydeg = 3, element_type = Tri(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralWeakForm())

top_bottom(x, tol=50*eps()) = abs(abs(x[2]) - 1) < tol
is_on_boundary = Dict(:top_bottom => top_bottom)
mesh = DGMultiMesh(dg, cells_per_dimension=(16, 16); periodicity=(true, false), is_on_boundary)

initial_condition = initial_condition_navier_stokes_convergence_test

# BC types
velocity_bc_top_bottom = NoSlip((x, t, equations) -> initial_condition_navier_stokes_convergence_test(x, t, equations)[2:3])
heat_bc_top_bottom = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_top_bottom = BoundaryConditionViscousWall(velocity_bc_top_bottom, heat_bc_top_bottom)

# define inviscid boundary conditions
boundary_conditions = (; :top_bottom => boundary_condition_slip_wall)

# define viscous boundary conditions
boundary_conditions_parabolic = (; :top_bottom => boundary_condition_top_bottom)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, dg;
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
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback, alive_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(), abstol=time_int_tol, reltol=time_int_tol,
            save_everystep=false, callback=callbacks)
summary_callback() # print the timer summary
