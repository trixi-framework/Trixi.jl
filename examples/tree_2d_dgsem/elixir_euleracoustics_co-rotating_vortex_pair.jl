using LinearAlgebra: norm
using OrdinaryDiffEqLowStorageRK
using Trixi

# This elixir calculates the sound generated by a pair of co-rotating vortices in a quiescent
# medium.
# Note that this elixir can take several hours to run. For optimal results consider increasing the
# initial_refinement_level to 8.
#
# References:
# - Duck Joo Lee, Sam Ok Koo (1995)
#   Numerical study of sound generation due to a spinning vortex
#   https://doi.org/10.2514/3.12327
# - Michael Schlottke-Lakemper (2017)
#   A direct-hybrid method for aeroacoustic analysis
#   https://doi.org/10.18154/RWTH-2017-04082
module VortexPairSetup

using LinearAlgebra: norm
using Trixi

# Parameters that describe the co-rotating vortex pair
struct VortexPair{RealT <: Real}
    r0::RealT # Distance between origin and each vortex center
    rc::RealT # Vortex core radius
    c0::RealT # Speed of sound
    circulation::RealT # Circulation of the vortices
    rho0::RealT # Density
end

# Analytical flow solution, used for the initial condition of the flow simulation
function velocity(x, t, vortex_pair::VortexPair)
    RealT = eltype(x)
    @unpack r0, rc, circulation = vortex_pair

    omega = circulation / (4 * convert(RealT, pi) * r0^2)
    si, co = sincos(omega * t)
    b = SVector(r0 * co, r0 * si) # vortex centers are b and -b
    z_plus = x - b
    z_minus = x + b

    # Transform to polar coordinates
    r_plus = norm(z_plus)
    r_minus = norm(z_minus)
    theta_plus = atan(z_plus[2], z_plus[1])
    theta_minus = atan(z_minus[2], z_minus[1])

    si_plus, co_plus = sincos(theta_plus)
    si_minus, co_minus = sincos(theta_minus)

    v1 = -circulation / (2 * convert(RealT, pi)) * (r_plus / (rc^2 + r_plus^2) * si_plus +
          r_minus / (rc^2 + r_minus^2) * si_minus)
    v2 = circulation / (2 * convert(RealT, pi)) * (r_plus / (rc^2 + r_plus^2) * co_plus +
          r_minus / (rc^2 + r_minus^2) * co_minus)

    return SVector(v1, v2)
end

# Initial condition of the flow simulation. Uses constant density rho0 and analytical velocities.
# The pressure is calculated using the given speed of sound c0 and Bernoulli's principle
struct InitialCondition{RealT <: Real}
    vortex_pair::VortexPair{RealT}
end

function (initial_condition::InitialCondition)(x, t,
                                               equations::CompressibleEulerEquations2D)
    @unpack vortex_pair = initial_condition
    @unpack rho0, c0 = vortex_pair
    gamma = equations.gamma

    v = velocity(x, t, vortex_pair)
    p0 = rho0 * c0^2 / gamma
    p = p0 - 0.5f0 * (gamma - 1) / gamma * sum(v .^ 2) # Bernoulli's principle

    prim = SVector(rho0, v[1], v[2], p)
    return prim2cons(prim, equations)
end

# For both the flow and acoustics solvers, a sponge layer is used to dampen the density
# and pressure towards the freestream values (for the flow solver) and the perturbed pressure
# to zero (for the acoustics solver).
struct SpongeLayer{RealT <: Real, uEltype <: Real, N, SourceTerms}
    sponge_layer_min::NTuple{4, RealT} # (-x,+x,-y,+y) min coordinates of sponge layer per direction
    sponge_layer_max::NTuple{4, RealT} # (-x,+x,-y,+y) max coordinates of sponge layer per direction
    reference_values::NTuple{N, uEltype} # reference values for variables affected by sponge layer
    source_terms::SourceTerms # source terms to be used outside the sponge zone
end

function SpongeLayer(; sponge_layer_min, sponge_layer_max, reference_values,
                     source_terms = nothing)
    return SpongeLayer(sponge_layer_min, sponge_layer_max, reference_values, source_terms)
end

function (sponge_layer::SpongeLayer)(u, x, t, equations)
    @unpack sponge_layer_min, sponge_layer_max, reference_values, source_terms = sponge_layer

    if lies_in_sponge_layer(x, sponge_layer_min, sponge_layer_max)
        return source_term_sponge_layer(u, x, t, equations, sponge_layer_min,
                                        sponge_layer_max,
                                        reference_values)
    elseif source_terms !== nothing
        return source_terms(u, x, t, equations)
    else
        return SVector(ntuple(v -> zero(eltype(u)), Val(nvariables(equations))))
    end
end

function lies_in_sponge_layer(x, sponge_layer_min, sponge_layer_max)
    return (sponge_layer_min[1] <= x[1] <= sponge_layer_max[1]) || # -x direction
           (sponge_layer_min[2] <= x[1] <= sponge_layer_max[2]) || # +x direction
           (sponge_layer_min[3] <= x[2] <= sponge_layer_max[3]) || # -y direction
           (sponge_layer_min[4] <= x[2] <= sponge_layer_max[4])    # +y direction
end

function source_term_sponge_layer(u, x, t, equations::AcousticPerturbationEquations2D,
                                  sponge_layer_min::NTuple{4}, sponge_layer_max::NTuple{4},
                                  reference_values)
    # Perturbed pressure source is -alpha^2 * (u - reference_value) where alpha in [0,1] is a damping
    # factor depending on the position inside the sponge layer

    # Calculate the damping factors for each direction (=0 if x is not in the corresponding sponge
    # zone) and take the maximum. This ensures proper damping if x lies in a corner where the sponge
    # zones for two directions overlap
    alphas = ntuple(i -> calc_damping_factor(x, i, sponge_layer_min, sponge_layer_max),
                    Val(2 * ndims(equations)))
    alpha_square = maximum(alphas)^2

    return SVector(0, 0, -alpha_square * (u[3] - reference_values[1] / u[6]^2), 0, 0, 0, 0)
end

function source_term_sponge_layer(u, x, t, equations::CompressibleEulerEquations2D,
                                  sponge_layer_min::NTuple{4}, sponge_layer_max::NTuple{4},
                                  reference_values)
    # Calculate the damping factors for each direction (=0 if x is not in the corresponding sponge
    # zone) and take the maximum. This ensures proper damping if x lies in a corner where the sponge
    # zones for two directions overlap
    alphas = ntuple(i -> calc_damping_factor(x, i, sponge_layer_min, sponge_layer_max),
                    Val(2 * ndims(equations)))
    alpha_square = maximum(alphas)^2

    u_prim = cons2prim(u, equations)
    s = SVector(-alpha_square * (u_prim[1] - reference_values[1]), 0, 0,
                -alpha_square * (u_prim[4] - reference_values[2]))

    return prim2cons(s, equations)
end

function calc_damping_factor(x, direction, sponge_layer_min, sponge_layer_max)
    # Damping factor alpha grows linearly from 0 to 1 depending on how deep x lies in the sponge layer
    # If x does not lie in the sponge layer, this returns 0

    # Get the coordinate that determines how deep we are in the sponge zone
    if direction in (1, 2)
        pos = x[1]
    else
        pos = x[2]
    end

    # Determine where the sponge layer begins/ends to allow calculating the damping factor
    if iseven(direction)
        sponge_begin = sponge_layer_min[direction]
        sponge_end = sponge_layer_max[direction]
    else
        sponge_begin = sponge_layer_max[direction]
        sponge_end = sponge_layer_min[direction]
    end

    alpha = (pos - sponge_begin) / (sponge_end - sponge_begin)

    # alpha lies in [0, 1] if and only if x lies in the sponge zone
    if 0 <= alpha <= 1
        return alpha
    else
        return zero(alpha)
    end
end

# Boundary condition for the flow problem: The sponge layer dampens density and pressure towards the
# freestream values. The freestream values (converted into conservative variables) are therefore
# used as a Dirichlet boundary
struct BoundaryCondition{uEltype}
    rho::uEltype
    rho_e::uEltype
end

function (bc::BoundaryCondition)(u_inner, orientation, direction, x, t,
                                 surface_flux_function,
                                 equations::CompressibleEulerEquations2D)
    u_boundary = SVector(bc.rho, zero(bc.rho), zero(bc.rho), bc.rho_e)

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end

end # module

import .VortexPairSetup

###############################################################################
# shared parameters, mesh and solver for both semidiscretizations

# Parameters of the vortex pair
Mach = 1 / 9
c0 = 1.0
r0 = 1.0
circulation = 4 * pi * r0 * c0 * Mach
rho = 1.0

rc = 2 / 9 * r0 * 1.0

T_r = 8 * pi^2 * r0^2 / circulation # Rotational period of the vortex pair
T_a = T_r / 2 # Acoustic period of the vortex pair

vortex_pair = VortexPairSetup.VortexPair(r0, rc, c0, circulation, rho)

# Shared mesh for both semidiscretizations
coordinates_min = (-135 * r0, -135 * r0) # minimum coordinates (min(x), min(y))
coordinates_max = (135 * r0, 135 * r0) # maximum coordinates (max(x), max(y))
refinement_patches = ((type = "sphere", center = (0.0, 0.0), radius = 85.0 * r0),
                      (type = "sphere", center = (0.0, 0.0), radius = 20.0 * r0),
                      (type = "sphere", center = (0.0, 0.0), radius = 10.0 * r0),
                      (type = "sphere", center = (0.0, 0.0), radius = 5.0 * r0))
initial_refinement_level = 7
n_cells_max = 500_000
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                refinement_patches = refinement_patches,
                n_cells_max = n_cells_max, # set maximum capacity of tree data structure
                periodicity = false)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
solver = DGSEM(polydeg = 3, surface_flux = FluxLaxFriedrichs(max_abs_speed_naive))

###############################################################################
# semidiscretization Euler equations

gamma = 1.4
equations_euler = CompressibleEulerEquations2D(gamma)

initial_condition_euler = VortexPairSetup.InitialCondition(vortex_pair)

sponge_layer_euler = VortexPairSetup.SpongeLayer(sponge_layer_min = (-135 * r0, 115 * r0,
                                                                     -135 * r0, 115 * r0),
                                                 sponge_layer_max = (-115 * r0, 135 * r0,
                                                                     -115 * r0, 135 * r0),
                                                 reference_values = (rho,
                                                                     rho * c0^2 / gamma)) # (rho0, p0)

boundary_condition_euler = VortexPairSetup.BoundaryCondition(rho,
                                                             (rho * c0^2 / gamma) /
                                                             (gamma - 1))

semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition_euler,
                                          solver,
                                          boundary_conditions = boundary_condition_euler,
                                          source_terms = sponge_layer_euler)

###############################################################################
# semidiscretization acoustic perturbation equations
equations_acoustics = AcousticPerturbationEquations2D(v_mean_global = (13.0, 26.0),
                                                      c_mean_global = 39.0,
                                                      rho_mean_global = 52.0) # global mean values will be overwritten

sponge_layer_acoustics = VortexPairSetup.SpongeLayer(sponge_layer_min = (-135 * r0,
                                                                         100 * r0,
                                                                         -135 * r0,
                                                                         100 * r0),
                                                     sponge_layer_max = (-100 * r0,
                                                                         135 * r0,
                                                                         -100 * r0,
                                                                         135 * r0),
                                                     reference_values = (0.0,))

"""
    boundary_condition_zero(u_inner, orientation, direction, x, t, surface_flux_function,
                            equations::AcousticPerturbationEquations2D)

Boundary condition that uses a boundary state where the state variables are zero and the mean
variables are the same as in `u_inner`.
"""
function boundary_condition_zero(u_inner, orientation, direction, x, t,
                                 surface_flux_function,
                                 equations::AcousticPerturbationEquations2D)
    value = zero(eltype(u_inner))
    u_boundary = SVector(value, value, value, cons2mean(u_inner, equations)...)

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end

semi_acoustics = SemidiscretizationHyperbolic(mesh, equations_acoustics,
                                              initial_condition_constant,
                                              solver,
                                              boundary_conditions = boundary_condition_zero,
                                              source_terms = sponge_layer_acoustics)

###############################################################################
# ODE solvers, callbacks etc. for averaging the flow field

# Create ODE problem
tspan1 = (0.0, 400.0)
ode_averaging = semidiscretize(semi_euler, tspan1)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

analysis_interval = 5000
alive_callback = AliveCallback(analysis_interval = analysis_interval)

tspan_averaging = (50.0, 400.0)
averaging_callback = AveragingCallback(semi_euler, tspan_averaging)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks_averaging = CallbackSet(summary_callback, alive_callback, averaging_callback,
                                  stepsize_callback)

###############################################################################
# run simulation for averaging the flow field

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol_averaging = solve(ode_averaging, CarpenterKennedy2N54(williamson_condition = false);
                      dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                      ode_default_options()..., callback = callbacks_averaging);

###############################################################################
# set up coupled semidiscretization

source_region(x) = sum(abs2, x) < 6^2 # calculate sources within radius 6 around origin
# gradually reduce acoustic source term amplitudes to zero, starting at radius 5
weights(x) = sum(abs2, x) < 5^2 ? one(eltype(x)) : cospi(0.5f0 * (norm(x) - 5))

semi = SemidiscretizationEulerAcoustics(semi_acoustics, semi_euler,
                                        source_region = source_region, weights = weights)

###############################################################################
# ODE solvers, callbacks etc. for the coupled simulation

# Create ODE problem
tspan = (0.0, 7.0 * T_a)
ode = semidiscretize(semi, tspan)
# We need an additional ODE for the pure flow problem
ode_euler = semidiscretize(semi.semi_euler, tspan)

# Set up coupling callback
cfl_acoustics = 0.8
cfl_euler = 0.8
euler_acoustics_coupling = EulerAcousticsCouplingCallback(ode_euler, "out/averaging.h5",
                                                          CarpenterKennedy2N54(williamson_condition = false),
                                                          cfl_acoustics, cfl_euler,
                                                          callback = SaveRestartCallback(interval = 2300,
                                                                                         output_directory = "out/euler/"))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

analysis_interval = 5000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

output_directory = "out/"
save_solution = SaveSolutionCallback(interval = 2300, output_directory = output_directory)
save_restart = SaveRestartCallback(interval = 2300, output_directory = output_directory)

callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback, save_solution,
                        save_restart, euler_acoustics_coupling)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
