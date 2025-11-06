using Trixi
using OrdinaryDiffEqBDF
using ADTypes ## This is needed to force 'autodiff = AutoFiniteDiff()' in the ODE solver.

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

    v_dot_n_avg_horizontal = v1_avg * normal_direction[1] + v2_avg * normal_direction[2]
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

coordinates_min = (0.0, 0.0)
coordinates_max = (20_000.0, 10_000.0)

trees_per_dimension = (16, 8)

mesh = P4estMesh(trees_per_dimension, polydeg = polydeg,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = (true, false), initial_refinement_level = 0)

boundary_conditions = Dict(:y_neg => boundary_condition_slip_wall_vel,
                           :y_pos => boundary_condition_slip_wall_vel)

initial_condition = initial_condition_warm_bubble

semi = SemidiscretizationHyperbolicSplit(mesh,
                                         (equations, equations),
                                         initial_condition,
                                         solver_implicit,
                                         solver_explicit;
                                         boundary_conditions = (boundary_conditions,
                                                                boundary_conditions),
                                         source_terms = (nothing, source_terms_gravity),)
dt = 0.5
tspan = (0.0, 1000.0)
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
#	    CarpenterKennedy2N54();
            dt = dt, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false,
            callback = callbacks,);
