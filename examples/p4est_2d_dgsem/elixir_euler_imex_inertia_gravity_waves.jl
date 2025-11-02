using Trixi
using OrdinaryDiffEqBDF
using SparseDiffTools ## This is needed to force 'autodiff = AutoFiniteDiff()' in the ODE solver.
using OrdinaryDiffEqLowOrderRK

function initial_condition_inertia_gravity_waves(x, t, equations::CompressibleEulerEquations2D)
	g = 9.81
	c_p = 1004.0
	c_v = 717.0

	A = 5000.0
	H = 10000
	b0 = 0.01
	xc = -150_000 + 100_000
	binv = (1 + (x[1] - xc)^2 / A^2)
	potential_temperature_ref = 300 * exp(0.01^2 / g * x[2])
	potential_temperature_perturbation = b0 * sin(pi * x[2] / H) / binv

	potential_temperature = potential_temperature_ref + potential_temperature_perturbation

	# Exner pressure, solves hydrostatic equation for x[2]
	exner = 1 + g^2 / (c_p * 300.0 * 0.01^2) * (exp(-0.01^2 / g * x[2]) - 1)

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

@inline function flux_lmars_fast(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CompressibleEulerEquations2D)
    a = 340.0
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    norm_ = Trixi.norm(normal_direction)

    rho = 0.5f0 * (rho_ll + rho_rr)

    p_interface = 0.5f0 * (p_ll + p_rr) - 0.5f0 * a * rho * (v_rr - v_ll) / norm_
    v_interface = 0.5f0 * (v_ll + v_rr) - 1 / (2 * a * rho) * (p_rr - p_ll) * norm_

    if (v_interface > 0)
	f4 = p_ll * v_interface
    else
	f4 = p_rr * v_interface
    end

	return SVector(zero(eltype(u_ll)),
        p_interface * normal_direction[1],
        p_interface * normal_direction[2],
	f4)
end

@inline function flux_lmars_slow(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CompressibleEulerEquations2D)
    a = 340.0
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    norm_ = Trixi.norm(normal_direction)

    rho = 0.5f0 * (rho_ll + rho_rr)

    p_interface = 0.5f0 * (p_ll + p_rr) - 0.5f0 * a * rho * (v_rr - v_ll) / norm_
    v_interface = 0.5f0 * (v_ll + v_rr) - 1 / (2 * a * rho) * (p_rr - p_ll) * norm_

    if (v_interface > 0)
        f1, f2, f3, f4 = u_ll * v_interface
    else
        f1, f2, f3, f4 = u_rr * v_interface
    end

    return SVector(f1,
        f2, 
        f3,
        f4)
end

@inline function flux_kennedy_gruber_slow(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::CompressibleEulerEquations2D)
    # Unpack left and right state
    rho_e_ll = last(u_ll)
    rho_e_rr = last(u_rr)
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    # Average each factor of products in flux
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v_dot_n_avg = v1_avg * normal_direction[1] + v2_avg * normal_direction[2]
    p_avg = 0.5f0 * (p_ll + p_rr)
    e_avg = 0.5f0 * (rho_e_ll / rho_ll + rho_e_rr / rho_rr)

    v_dot_n_avg_horizontal = v1_avg * normal_direction[1] 
    # Calculate fluxes depending on normal_direction
    f1 = rho_avg * v_dot_n_avg
    f2 = f1 * v1_avg 
    f3 = f1 * v2_avg 
    f4 = f1 * e_avg 

    return SVector(f1, f2, f3, f4)
end

@inline function flux_kennedy_gruber_fast(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::CompressibleEulerEquations2D)
    # Unpack left and right state
    rho_e_ll = last(u_ll)
    rho_e_rr = last(u_rr)
    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    # Average each factor of products in flux
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v_dot_n_avg = v1_avg * normal_direction[1] + v2_avg * normal_direction[2]
    p_avg = 0.5f0 * (p_ll + p_rr)

    # Calculate fluxes depending on normal_direction
    f2 = p_avg * normal_direction[1]
    f3 = p_avg * normal_direction[2]
    f4 = p_avg * v_dot_n_avg

	return SVector(zero(eltype(u_ll)), f2, f3, f4)
end

@inline function source_terms_gravity(u, x, t, equations::CompressibleEulerEquations2D)
    g = 9.81
    rho, _, rho_v2, _ = u
    return SVector(zero(eltype(u)), zero(eltype(u)), -g * rho, -g * rho_v2)
end

equations = CompressibleEulerEquations2D(1004 / 717)

polydeg = 2
basis = LobattoLegendreBasis(polydeg)

volume_integral_explicit = VolumeIntegralFluxDifferencing(flux_kennedy_gruber_slow)
solver_explicit = DGSEM(basis, flux_lmars_slow, volume_integral_explicit)

volume_integral_implicit = VolumeIntegralFluxDifferencing(flux_kennedy_gruber_fast)
solver_implicit = DGSEM(basis, flux_lmars_fast, volume_integral_implicit)

coordinates_min = (-150_000.0, 0.0)
coordinates_max = (150_000.0, 10_000.0)

trees_per_dimension = (40, 5)

mesh = P4estMesh(trees_per_dimension, polydeg = polydeg,
	coordinates_min = coordinates_min, coordinates_max = coordinates_max,
	periodicity = (true, false), initial_refinement_level = 0)

boundary_conditions_explicit = Dict(:y_neg => boundary_condition_slip_wall_vel,
                                    :y_pos => boundary_condition_slip_wall_vel)
boundary_conditions_implicit = Dict(:y_neg => boundary_condition_slip_wall_vel,
                                    :y_pos => boundary_condition_slip_wall_vel)

initial_condition = initial_condition_inertia_gravity_waves

semi = SemidiscretizationHyperbolicSplit(mesh,
                                         (equations, equations),
                                         initial_condition,
                                         solver_implicit,
                                         solver_explicit;
                                         boundary_conditions = (boundary_conditions_implicit,
                                                                boundary_conditions_explicit),
                                         source_terms = (nothing, source_terms_gravity),)
T = 3000.0
dt = 1.0
tspan = (0.0, 3000.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 1)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100, solution_variables = cons2prim)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution)

###############################################################################
# run the simulation
# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode,
           SBDF2(autodiff = AutoFiniteDiff());
          # Euler();
            dt = dt, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false,
            callback = callbacks,);
