using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg = 3, element_type = Tri(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralWeakForm())

equations = LinearScalarAdvectionEquation2D(1.5, 1.0)
equations_parabolic = LaplaceDiffusion2D(5e-2)

initial_condition_zero(x, t, equations::LinearScalarAdvectionEquation2D) = SVector(0.0)
initial_condition = initial_condition_zero

# tag different boundary segments
left(x, tol=50*eps()) = abs(x[1] + 1) < tol
right(x, tol=50*eps()) = abs(x[1] - 1) < tol
bottom(x, tol=50*eps()) = abs(x[2] + 1) < tol
top(x, tol=50*eps()) = abs(x[2] - 1) < tol
is_on_boundary = Dict(:left => left, :right => right, :top => top, :bottom => bottom)
mesh = DGMultiMesh(dg, cells_per_dimension=(16, 16); is_on_boundary)

# BC types
boundary_condition_left = BoundaryConditionDirichlet((x, t, equations) -> SVector(1 + 0.1 * x[2]))
boundary_condition_zero = BoundaryConditionDirichlet((x, t, equations) -> SVector(0.0))
boundary_condition_neumann_zero = BoundaryConditionNeumann((x, t, equations) -> SVector(0.0))

# define inviscid boundary conditions
boundary_conditions = (; :left => boundary_condition_left,
                         :bottom => boundary_condition_zero,
                         :top => boundary_condition_do_nothing,
                         :right => boundary_condition_do_nothing)

# define viscous boundary conditions
parabolic_boundary_conditions = (; :left => boundary_condition_left,
                                   :bottom => boundary_condition_zero,
                                   :top => boundary_condition_zero,
                                   :right => boundary_condition_neumann_zero)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, dg;
                                             boundary_conditions=(boundary_conditions, parabolic_boundary_conditions))

tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
callbacks = CallbackSet(summary_callback, alive_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-6
saveat = LinRange(tspan..., 10)
sol = solve(ode, RDPK3SpFSAL49(), abstol=time_int_tol, reltol=time_int_tol, save_everystep=false,
            saveat=saveat, callback=callbacks)
summary_callback() # print the timer summary
