
using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg = 3, element_type = Tri(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralWeakForm())

equations = LinearScalarAdvectionEquation2D(1.0, 1.0)
parabolic_equations = ScalarDiffusion2D(1e-3)

initial_condition_zero(x, t, equations::LinearScalarAdvectionEquation2D) = SVector(0.0)
initial_condition = initial_condition_zero

# tag two separate boundary segments
inflow(x, tol=50*eps()) = abs(x[1]+1)<tol || abs(x[2]+1)<tol
rest_of_boundary(x, tol=50*eps()) = !inflow(x, tol)
is_on_boundary = Dict(:inflow => inflow, :outflow => rest_of_boundary)
mesh = DGMultiMesh(dg, cells_per_dimension=(16, 16), is_on_boundary=is_on_boundary)

# define inviscid boundary conditions
boundary_condition_inflow = BoundaryConditionDirichlet((x, t, equations) -> SVector(1.0))
boundary_conditions = (; :inflow => boundary_condition_inflow)

# define viscous boundary conditions
boundary_condition_wall = BoundaryConditionDirichlet((x, t, equations) -> SVector(0.0))
parabolic_boundary_conditions = (; :inflow => boundary_condition_inflow,
                                   :outflow => boundary_condition_wall)

semi = SemidiscretizationHyperbolicParabolic(mesh, equations, parabolic_equations, initial_condition, dg;
                                             boundary_conditions, parabolic_boundary_conditions)

tspan = (0.0, .5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
callbacks = CallbackSet(summary_callback, alive_callback)

###############################################################################
# run the simulation

tol = 1e-6
tsave = LinRange(tspan..., 10)
sol = solve(ode, RDPK3SpFSAL49(), abstol=tol, reltol=tol, save_everystep=false,
            saveat=tsave, callback=callbacks)
summary_callback() # print the timer summary
