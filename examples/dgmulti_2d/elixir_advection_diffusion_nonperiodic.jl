using Trixi, OrdinaryDiffEq

polydeg = 3
r1D, w1D = StartUpDG.gauss_lobatto_quad(0, 0, polydeg)
rq, sq = vec.(StartUpDG.NodesAndModes.meshgrid(r1D, r1D))
wq = (x->x[1] .* x[2])(vec.(StartUpDG.NodesAndModes.meshgrid(w1D, w1D)))

dg = DGMulti(polydeg = polydeg, element_type = Quad(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralWeakForm();
             quad_rule_vol=(rq,sq,wq), quad_rule_face=(r1D,w1D))

get_diffusivity() = 5.0e-2

equations = LinearScalarAdvectionEquation2D(1.0, 0.0)
equations_parabolic = LaplaceDiffusion2D(get_diffusivity(), equations)

# from "Robust DPG methods for transient convection-diffusion."
# Building bridges: connections and challenges in modern approaches to numerical partial differential equations.
# Springer, Cham, 2016. 179-203. Ellis, Truman, Jesse Chan, and Leszek Demkowicz."
function initial_condition_erikkson_johnson(x, t, equations)
  l = 4
  epsilon = get_diffusivity() # TODO: this requires epsilon < .6 due to the sqrt
  lambda_1 = (-1 + sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
  lambda_2 = (-1 - sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
  r1 = (1 + sqrt(1 + 4 * pi^2 * epsilon^2)) / (2 * epsilon)
  s1 = (1 - sqrt(1 + 4 * pi^2 * epsilon^2)) / (2 * epsilon)
  u = exp(-l * t) * (exp(lambda_1 * x[1]) - exp(lambda_2 * x[1])) +
      cos(pi * x[2]) * (exp(s1 * x[1]) - exp(r1 * x[1])) / (exp(-s1) - exp(-r1))
  return SVector{1}(u)
end
initial_condition = initial_condition_erikkson_johnson

# tag different boundary segments
left(x, tol=50*eps()) = abs(x[1] + 1) < tol
right(x, tol=50*eps()) = abs(x[1]) < tol
bottom(x, tol=50*eps()) = abs(x[2] + .5) < tol
top(x, tol=50*eps()) = abs(x[2] - .5) < tol
entire_boundary(x, tol=50*eps()) = true
is_on_boundary = Dict(:left => left, :right => right, :top => top, :bottom => bottom,
                      :entire_boundary => entire_boundary)
mesh = DGMultiMesh(dg; coordinates_min=(-1.0, -0.5), coordinates_max=(0.0, 0.5),
                   cells_per_dimension=(16, 16), is_on_boundary)

# BC types
boundary_condition = BoundaryConditionDirichlet(initial_condition)

# define inviscid boundary conditions
boundary_conditions = (; :left   => boundary_condition,
                         :top    => boundary_condition,
                         :bottom => boundary_condition,
                         :right  => boundary_condition_do_nothing)

# define viscous boundary conditions
boundary_conditions_parabolic = (; :entire_boundary => boundary_condition)

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

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(), abstol=time_int_tol, reltol=time_int_tol,
            save_everystep=false, callback=callbacks)
summary_callback() # print the timer summary
