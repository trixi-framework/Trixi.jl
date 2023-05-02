using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg = 3, element_type = Tri(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralWeakForm())

equations = LinearScalarAdvectionEquation2D(1.5, 1.0)
equations_parabolic = LaplaceDiffusion2D(5.0e-2, equations)

initial_condition_zero(x, t, equations::LinearScalarAdvectionEquation2D) = SVector(0.0)
initial_condition = initial_condition_zero

# tag different boundary segments
left(x, tol=50*eps()) = abs(x[1] + 1) < tol
right(x, tol=50*eps()) = abs(x[1] - 1) < tol
bottom(x, tol=50*eps()) = abs(x[2] + 1) < tol
top(x, tol=50*eps()) = abs(x[2] - 1) < tol
is_on_boundary = Dict(:left => left, :right => right, :top => top, :bottom => bottom)

cells_per_dimension = (16, 16)
mesh = DGMultiMesh(dg, cells_per_dimension; is_on_boundary)

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
boundary_conditions_parabolic = (; :left => boundary_condition_left,
                                   :bottom => boundary_condition_zero,
                                   :top => boundary_condition_zero,
                                   :right => boundary_condition_neumann_zero)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, dg;
                                             boundary_conditions=(boundary_conditions, boundary_conditions_parabolic))

tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback, alive_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-6
sol = solve(ode, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol,
            ode_default_options()..., callback=callbacks)
summary_callback() # print the timer summary
