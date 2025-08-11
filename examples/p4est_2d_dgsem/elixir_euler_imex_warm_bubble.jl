using Trixi
using OrdinaryDiffEqBDF
using OrdinaryDiffEqIMEXMultistep
using SparseDiffTools ## This is needed to force 'autodiff = AutoFiniteDiff()' in the ODE solver.

function initial_condition_warm_bubble(x, t, equations::CompressibleEulerEquations2D)
	g = 9.81
	c_p = 1004.0
	c_v = 717.0

	# center of perturbation
	center_x = 10000.0
	center_z = 2000.0
	# radius of perturbation
	radius = 2000.0
	# distance of current x to center of perturbation
	r = sqrt((x[1] - center_x)^2 + (x[2] - center_z)^2)

	# perturbation in potential temperature
	potential_temperature_ref = 300.0
	potential_temperature_perturbation = 0.0
	if r <= radius
		potential_temperature_perturbation = 2 * cospi(0.5 * r / radius)^2
	end
	potential_temperature = potential_temperature_ref + potential_temperature_perturbation

	# Exner pressure, solves hydrostatic equation for x[2]
	exner = 1 - g / (c_p * potential_temperature) * x[2]

	# pressure
	p_0 = 100_000.0  # reference pressure
	R = c_p - c_v    # gas constant (dry air)
	p = p_0 * exner^(c_p / R)

	# temperature
	T = potential_temperature * exner

	# density
	rho = p / (R * T)

	v1 = 20.0
	v2 = 0.0
	return prim2cons(SVector(rho, v1, v2, p), equations)
end

@inline function boundary_condition_slip_wall_vel(u_inner, normal_direction::AbstractVector,
	x, t,
	surface_flux_function,
	equations::CompressibleEulerEquations2D)
	# normalize the outward pointing direction
	normal = normal_direction / Trixi.norm(normal_direction)

	# compute the normal velocity
	u_normal = normal[1] * u_inner[2] + normal[2] * u_inner[3]

	# create the "external" boundary solution state
	u_boundary = SVector(u_inner[1],
		u_inner[2] - 2 * u_normal * normal[1],
		u_inner[3] - 2 * u_normal * normal[2],
		u_inner[4])

	# calculate the boundary flux
	flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

	return flux
end

@inline function boundary_condition_zero(u_inner, normal_direction::AbstractVector,
	x, t,
	surface_flux_function,
	equations::CompressibleEulerEquations2D)

	flux = flux_zero(u_inner, u_inner, normal_direction, equations)

	return flux
end

@inline function flux_zero(u_ll, u_rr, normal_direction,
	equations::CompressibleEulerEquations2D)
	return SVector(0.0, 0.0, 0.0, 0.0)
end

@inline function source_terms_gravity(u, x, t, equations::CompressibleEulerEquations2D)
	g = 9.81
	rho, _, rho_v2, _ = u
	return SVector(zero(eltype(u)), zero(eltype(u)), -g * rho, -g * rho_v2)
end

equations = CompressibleEulerEquations2D(1004 / 717)

polydeg = 2
basis = LobattoLegendreBasis(polydeg)

volume_integral_explicit = VolumeIntegralFluxDifferencing(flux_kennedy_gruber)
solver_explicit = DGSEM(basis, FluxLMARS(340.0), volume_integral_explicit)

volume_integral_implicit = VolumeIntegralFluxDifferencing(flux_zero)
solver_implicit = DGSEM(basis, flux_zero, volume_integral_implicit)

coordinates_min = (0.0, 0.0)
coordinates_max = (20_000.0, 10_000.0)

trees_per_dimension = (16, 16)

mesh = P4estMesh(trees_per_dimension, polydeg = polydeg,
	coordinates_min = coordinates_min, coordinates_max = coordinates_max,
	periodicity = (true, false), initial_refinement_level = 0)

boundary_conditions_explicit = Dict(:y_neg => boundary_condition_slip_wall_vel,
	:y_pos => boundary_condition_slip_wall_vel)
boundary_conditions_implicit = Dict(:y_neg => boundary_condition_zero,
	:y_pos => boundary_condition_zero)

initial_condition = initial_condition_warm_bubble

semi = SemidiscretizationHyperbolicSplit(
	mesh,
	(equations, equations),
	initial_condition,
	solver_implicit,
	solver_explicit;
	boundary_conditions = (boundary_conditions_implicit, boundary_conditions_explicit),
	source_terms = (source_terms_gravity, nothing),
)
T = 10.0
tspan = (0.0, T)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100, solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.5)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)

###############################################################################
# run the simulation
# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(
	ode,
	SBDF2(autodiff = AutoFiniteDiff());
	dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
	save_everystep = false,
	callback = callbacks,
);
