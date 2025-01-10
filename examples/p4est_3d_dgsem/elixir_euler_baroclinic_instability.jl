# An idealized baroclinic instability test case
# For optimal results consider increasing the resolution to 16x16x8 trees per cube face.
#
# Note that this elixir can take several hours to run.
# Using 24 threads of an AMD Ryzen Threadripper 3990X (more threads don't speed it up further)
# and `check-bounds=no`, this elixirs takes about one hour to run.
# With 16x16x8 trees per cube face on the same machine, it takes about 28 hours.
#
# References:
# - Paul A. Ullrich, Thomas Melvin, Christiane Jablonowski, Andrew Staniforth (2013)
#   A proposed baroclinic wave test case for deep- and shallow-atmosphere dynamical cores
#   https://doi.org/10.1002/qj.2241

using OrdinaryDiffEq
using Trixi
using LinearAlgebra

###############################################################################
# Setup for the baroclinic instability test
gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)

# Initial condition for an idealized baroclinic instability test
# https://doi.org/10.1002/qj.2241, Section 3.2 and Appendix A
function initial_condition_baroclinic_instability(x, t,
                                                  equations::CompressibleEulerEquations3D)
    lon, lat, r = cartesian_to_sphere(x)
    radius_earth = 6.371229e6
    # Make sure that the r is not smaller than radius_earth
    z = max(r - radius_earth, 0.0)

    # Unperturbed basic state
    rho, u, p = basic_state_baroclinic_instability_longitudinal_velocity(lon, lat, z)

    # Stream function type perturbation
    u_perturbation, v_perturbation = perturbation_stream_function(lon, lat, z)

    u += u_perturbation
    v = v_perturbation

    # Convert spherical velocity to Cartesian
    v1 = -sin(lon) * u - sin(lat) * cos(lon) * v
    v2 = cos(lon) * u - sin(lat) * sin(lon) * v
    v3 = cos(lat) * v

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

# Steady state for RHS correction below
function steady_state_baroclinic_instability(x, t, equations::CompressibleEulerEquations3D)
    lon, lat, r = cartesian_to_sphere(x)
    radius_earth = 6.371229e6
    # Make sure that the r is not smaller than radius_earth
    z = max(r - radius_earth, 0.0)

    # Unperturbed basic state
    rho, u, p = basic_state_baroclinic_instability_longitudinal_velocity(lon, lat, z)

    # Convert spherical velocity to Cartesian
    v1 = -sin(lon) * u
    v2 = cos(lon) * u
    v3 = 0.0

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

function cartesian_to_sphere(x)
    r = norm(x)
    lambda = atan(x[2], x[1])
    if lambda < 0
        lambda += 2 * pi
    end
    phi = asin(x[3] / r)

    return lambda, phi, r
end

# Unperturbed balanced steady-state.
# Returns primitive variables with only the velocity in longitudinal direction (rho, u, p).
# The other velocity components are zero.
function basic_state_baroclinic_instability_longitudinal_velocity(lon, lat, z)
    # Parameters from Table 1 in the paper
    # Corresponding names in the paper are commented
    radius_earth = 6.371229e6  # a
    half_width_parameter = 2           # b
    gravitational_acceleration = 9.80616     # g
    k = 3           # k
    surface_pressure = 1e5         # p₀
    gas_constant = 287         # R
    surface_equatorial_temperature = 310.0       # T₀ᴱ
    surface_polar_temperature = 240.0       # T₀ᴾ
    lapse_rate = 0.005       # Γ
    angular_velocity = 7.29212e-5  # Ω

    # Distance to the center of the Earth
    r = z + radius_earth

    # In the paper: T₀
    temperature0 = 0.5 * (surface_equatorial_temperature + surface_polar_temperature)
    # In the paper: A, B, C, H
    const_a = 1 / lapse_rate
    const_b = (temperature0 - surface_polar_temperature) /
              (temperature0 * surface_polar_temperature)
    const_c = 0.5 * (k + 2) * (surface_equatorial_temperature - surface_polar_temperature) /
              (surface_equatorial_temperature * surface_polar_temperature)
    const_h = gas_constant * temperature0 / gravitational_acceleration

    # In the paper: (r - a) / bH
    scaled_z = z / (half_width_parameter * const_h)

    # Temporary variables
    temp1 = exp(lapse_rate / temperature0 * z)
    temp2 = exp(-scaled_z^2)

    # In the paper: ̃τ₁, ̃τ₂
    tau1 = const_a * lapse_rate / temperature0 * temp1 +
           const_b * (1 - 2 * scaled_z^2) * temp2
    tau2 = const_c * (1 - 2 * scaled_z^2) * temp2

    # In the paper: ∫τ₁(r') dr', ∫τ₂(r') dr'
    inttau1 = const_a * (temp1 - 1) + const_b * z * temp2
    inttau2 = const_c * z * temp2

    # Temporary variables
    temp3 = r / radius_earth * cos(lat)
    temp4 = temp3^k - k / (k + 2) * temp3^(k + 2)

    # In the paper: T
    temperature = 1 / ((r / radius_earth)^2 * (tau1 - tau2 * temp4))

    # In the paper: U, u (zonal wind, first component of spherical velocity)
    big_u = gravitational_acceleration / radius_earth * k * temperature * inttau2 *
            (temp3^(k - 1) - temp3^(k + 1))
    temp5 = radius_earth * cos(lat)
    u = -angular_velocity * temp5 + sqrt(angular_velocity^2 * temp5^2 + temp5 * big_u)

    # Hydrostatic pressure
    p = surface_pressure *
        exp(-gravitational_acceleration / gas_constant * (inttau1 - inttau2 * temp4))

    # Density (via ideal gas law)
    rho = p / (gas_constant * temperature)

    return rho, u, p
end

# Perturbation as in Equations 25 and 26 of the paper (analytical derivative)
function perturbation_stream_function(lon, lat, z)
    # Parameters from Table 1 in the paper
    # Corresponding names in the paper are commented
    perturbation_radius = 1 / 6      # d₀ / a
    perturbed_wind_amplitude = 1.0      # Vₚ
    perturbation_lon = pi / 9     # Longitude of perturbation location
    perturbation_lat = 2 * pi / 9 # Latitude of perturbation location
    pertz = 15000    # Perturbation height cap

    # Great circle distance (d in the paper) divided by a (radius of the Earth)
    # because we never actually need d without dividing by a
    great_circle_distance_by_a = acos(sin(perturbation_lat) * sin(lat) +
                                      cos(perturbation_lat) * cos(lat) *
                                      cos(lon - perturbation_lon))

    # In the first case, the vertical taper function is per definition zero.
    # In the second case, the stream function is per definition zero.
    if z > pertz || great_circle_distance_by_a > perturbation_radius
        return 0.0, 0.0
    end

    # Vertical tapering of stream function
    perttaper = 1.0 - 3 * z^2 / pertz^2 + 2 * z^3 / pertz^3

    # sin/cos(pi * d / (2 * d_0)) in the paper
    sin_, cos_ = sincos(0.5 * pi * great_circle_distance_by_a / perturbation_radius)

    # Common factor for both u and v
    factor = 16 / (3 * sqrt(3)) * perturbed_wind_amplitude * perttaper * cos_^3 * sin_

    u_perturbation = -factor * (-sin(perturbation_lat) * cos(lat) +
                      cos(perturbation_lat) * sin(lat) * cos(lon - perturbation_lon)) /
                     sin(great_circle_distance_by_a)

    v_perturbation = factor * cos(perturbation_lat) * sin(lon - perturbation_lon) /
                     sin(great_circle_distance_by_a)

    return u_perturbation, v_perturbation
end

@inline function source_terms_baroclinic_instability(u, x, t,
                                                     equations::CompressibleEulerEquations3D)
    radius_earth = 6.371229e6  # a
    gravitational_acceleration = 9.80616     # g
    angular_velocity = 7.29212e-5  # Ω

    r = norm(x)
    # Make sure that r is not smaller than radius_earth
    z = max(r - radius_earth, 0.0)
    r = z + radius_earth

    du1 = zero(eltype(u))

    # Gravity term
    temp = -gravitational_acceleration * radius_earth^2 / r^3
    du2 = temp * u[1] * x[1]
    du3 = temp * u[1] * x[2]
    du4 = temp * u[1] * x[3]
    du5 = temp * (u[2] * x[1] + u[3] * x[2] + u[4] * x[3])

    # Coriolis term, -2Ω × ρv = -2 * angular_velocity * (0, 0, 1) × u[2:4]
    du2 -= -2 * angular_velocity * u[3]
    du3 -= 2 * angular_velocity * u[2]

    return SVector(du1, du2, du3, du4, du5)
end

###############################################################################
# Start of the actual elixir, semidiscretization of the problem

initial_condition = initial_condition_baroclinic_instability

boundary_conditions = Dict(:inside => boundary_condition_slip_wall,
                           :outside => boundary_condition_slip_wall)

# This is a good estimate for the speed of sound in this example.
# Other values between 300 and 400 should work as well.
surface_flux = FluxLMARS(340)
volume_flux = flux_kennedy_gruber
solver = DGSEM(polydeg = 5, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# For optimal results, use (16, 8) here
trees_per_cube_face = (8, 4)
mesh = Trixi.P4estMeshCubedSphere(trees_per_cube_face..., 6.371229e6, 30000.0,
                                  polydeg = 5, initial_refinement_level = 0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_baroclinic_instability,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10 * 24 * 60 * 60.0) # time in seconds for 10 days

# Save RHS of the steady state and subtract it in every RHS evaluation.
# This trick preserves the steady state exactly (to machine rounding errors, of course).
# Otherwise, this elixir produces entirely unusable results for a resolution of 8x8x4 cells
# per cube face with a polydeg of 3.
# With this trick, even the polydeg 3 simulation produces usable (although badly resolved) results,
# and most of the grid imprinting in higher polydeg simulation is eliminated.
#
# See https://github.com/trixi-framework/Trixi.jl/issues/980 for more information.
u_steady_state = compute_coefficients(steady_state_baroclinic_instability, tspan[1], semi)
# Use a `let` block for performance (otherwise du_steady_state will be a global variable)
let du_steady_state = similar(u_steady_state)
    # Save RHS of the steady state
    Trixi.rhs!(du_steady_state, u_steady_state, semi, tspan[1])

    global function corrected_rhs!(du, u, semi, t)
        # Normal RHS evaluation
        Trixi.rhs!(du, u, semi, t)
        # Correct by subtracting the steady-state RHS
        Trixi.@trixi_timeit Trixi.timer() "rhs correction" begin
            # Use Trixi.@threaded for threaded performance
            Trixi.@threaded for i in eachindex(du)
                du[i] -= du_steady_state[i]
            end
        end
    end
end
u0 = compute_coefficients(tspan[1], semi)
ode = ODEProblem(corrected_rhs!, u0, tspan, semi)

summary_callback = SummaryCallback()

analysis_interval = 5000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 5000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution)

###############################################################################
# run the simulation

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread = OrdinaryDiffEq.True()); abstol = 1.0e-6,
            reltol = 1.0e-6,
            ode_default_options()..., callback = callbacks);

summary_callback() # print the timer summary
