
using OrdinaryDiffEq
using LinearAlgebra: dot
using Trixi

###############################################################################
# semidiscretization of the 1 linearized Euler equations

rho_0 = 1.0
v_0 = 1.0
c_0 = 1.0
equations = LinearizedEulerEquations1D(rho_0, v_0, c_0)

solver = DGSEM(polydeg = 3, surface_flux = flux_hll)

coordinates_min = (0.0,) # minimum coordinate
coordinates_max = (1.0,) # maximum coordinate
cells_per_dimension = (64,)

mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

# Linearized Euler: Eigensystem
LinEuler_EigVals = [v_0 - c_0; v_0; v_0 + c_0]
LinEuler_EigVecs = [-rho_0/c_0 1 rho_0/c_0;
                    1 0 1;
                    -rho_0*c_0 0 rho_0*c_0]
LinEuler_EigVecs_inv = inv(LinEuler_EigVecs)

# Trace back characteristics. 
# See https://metaphor.ethz.ch/x/2019/hs/401-4671-00L/literature/mishra_hyperbolic_pdes.pdf, p.95
function compute_char_initial_pos(x, t)
    return SVector(x[1], x[1], x[1]) .- t * LinEuler_EigVals
end

function compute_primal_sol(char_vars)
    return LinEuler_EigVecs * char_vars
end

# Initial condition is in principle arbitrary, only periodicity is required
function initial_condition_entropy_wave(x, t, equations::LinearizedEulerEquations1D)
    # Parameters
    alpha = 1.0
    beta = 150.0
    center = 0.5

    rho_prime = alpha * exp(-beta * (x[1] - center)^2)
    v_prime = 0.0
    p_prime = 0.0

    return SVector(rho_prime, v_prime, p_prime)
end

function initial_condition_char_vars(x, t, equations::LinearizedEulerEquations1D)
    # Trace back characteristics
    x_char = compute_char_initial_pos(x, t)

    # Employ periodicity
    for p in 1:3
        while x_char[p] < coordinates_min[1]
            x_char[p] += coordinates_max[1] - coordinates_min[1]
        end
        while x_char[p] > coordinates_max[1]
            x_char[p] -= coordinates_max[1] - coordinates_min[1]
        end
    end

    # Set up characteristic variables
    w = zeros(3)
    for p in 1:3
        w[p] = dot(LinEuler_EigVecs_inv[p, :],
                   initial_condition_entropy_wave(x_char[p], 0, equations)) # Assumes t_0 = 0
    end

    return compute_primal_sol(w)
end

initial_condition = initial_condition_char_vars

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.3)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

stepsize_callback = StepsizeCallback(cfl = 1.0)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
