using OrdinaryDiffEqSSPRK
using Trixi
using LinearAlgebra: norm

###############################################################################
# Semidiscretizations of the polytropic Euler equations and Lattice-Boltzmann method (LBM)
# coupled using converter functions across their respective domains to generate a periodic system.
#
# In this elixir, we have a rectangular domain that is divided into a left and right half.
# On each half of the domain, an independent SemidiscretizationHyperbolic is created for each set of equations. 
# The two systems are coupled in the x-direction and are periodic in the y-direction.
# For a high-level overview, see also the figure below:
#
# (-2,  1)                                   ( 2,  1)
#     ┌────────────────────┬────────────────────┐
#     │    ↑ periodic ↑    │    ↑ periodic ↑    │
#     │                    │                    │
#     │     =========      │     =========      │
#     │     system #1      │     system #2      │
#     |       Euler        |        LBM         |
#     │     =========      │     =========      │
#     │                    │                    │
#     │<-- coupled         │<-- coupled         │
#     │         coupled -->│         coupled -->│
#     │                    │                    │
#     │    ↓ periodic ↓    │    ↓ periodic ↓    │
#     └────────────────────┴────────────────────┘
# (-2, -1)                                   ( 2, -1)

polydeg = 2
cells_per_dim_per_section = (16, 8)

###########
# system #1
# Euler
###########

### Setup taken from "elixir_eulerpolytropic_isothermal_wave.jl" ###

gamma = 1.0 # Isothermal gas
kappa = 1.0 # Scaling factor for the pressure, must fit to LBM `c_s`
eqs_euler = PolytropicEulerEquations2D(gamma, kappa)

volume_flux = flux_winters_etal
solver_euler = DGSEM(polydeg = polydeg, surface_flux = flux_hll,
                     volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# Linear pressure wave/Gaussian bump moving in the positive x-direction.
function initial_condition_pressure_bump(x, t, equations::PolytropicEulerEquations2D)
    rho = ((1.0 + 0.01 * exp(-(x[1] - 1)^2 / 0.1)) / equations.kappa)^(1 / equations.gamma)
    v1 = ((0.01 * exp(-(x[1] - 1)^2 / 0.1)) / equations.kappa)
    v2 = 0.0

    return prim2cons(SVector(rho, v1, v2), equations)
end
initial_condition_euler = initial_condition_pressure_bump

coords_min_euler = (-2.0, -1.0)
coords_max_euler = (0.0, 1.0)
mesh_euler = StructuredMesh(cells_per_dim_per_section,
                            coords_min_euler, coords_max_euler,
                            periodicity = (false, true))

# Use macroscopic variables derived from LBM populations 
# as boundary values for the Euler equations
function coupling_function_LBM2Euler(x, u, equations_other, equations_own)
    rho, v1, v2, _ = cons2macroscopic(u, equations_other)
    return prim2cons(SVector(rho, v1, v2), equations_own)
end

boundary_conditions_euler = (x_neg = BoundaryConditionCoupled(2, (:end, :i_forward),
                                                              Float64,
                                                              coupling_function_LBM2Euler),
                             x_pos = BoundaryConditionCoupled(2, (:begin, :i_forward),
                                                              Float64,
                                                              coupling_function_LBM2Euler),
                             y_neg = boundary_condition_periodic,
                             y_pos = boundary_condition_periodic)

semi_euler = SemidiscretizationHyperbolic(mesh_euler, eqs_euler, initial_condition_euler,
                                          solver_euler;
                                          boundary_conditions = boundary_conditions_euler)

###########
# system #2
# LBM
###########

# Results in c_s = c/sqrt(3) = 1. 
# This in turn implies that also in the LBM, p = c_s^2 * rho = 1 * rho = kappa * rho holds
# This is absolutely essential for the correct coupling between the two systems.
c = sqrt(3)

# Reference values `rho0, u0` correspond to the initial condition of the Euler equations.
# The gas should be inviscid (Re = Inf) to be consistent with the inviscid Euler equations.
# The Mach number `Ma` is computed internally from the speed of sound `c_s = c / sqrt(3)` and `u0`.
eqs_lbm = LatticeBoltzmannEquations2D(c = c, Re = Inf, rho0 = 1.0, u0 = 0.0, Ma = nothing)

# Quick & dirty implementation of the `flux_godunov` for Cartesian, yet structured meshes.
@inline function Trixi.flux_godunov(u_ll, u_rr, normal_direction::AbstractVector,
                                    equations::LatticeBoltzmannEquations2D)
    RealT = eltype(normal_direction)
    if isapprox(normal_direction[2], zero(RealT), atol = 2 * eps(RealT))
        v_alpha = equations.v_alpha1 * abs(normal_direction[1])
    elseif isapprox(normal_direction[1], zero(RealT), atol = 2 * eps(RealT))
        v_alpha = equations.v_alpha2 * abs(normal_direction[2])
    else
        error("Invalid normal direction for flux_godunov: $normal_direction")
    end
    return 0.5f0 * (v_alpha .* (u_ll + u_rr) - abs.(v_alpha) .* (u_rr - u_ll))
end

solver_lbm = DGSEM(polydeg = 2, surface_flux = flux_godunov)

function initial_condition_lbm(x, t, equations::LatticeBoltzmannEquations2D)
    rho = (1.0 + 0.01 * exp(-(x[1] - 1)^2 / 0.1))
    v1 = 0.01 * exp(-(x[1] - 1)^2 / 0.1)

    v2 = 0.0

    return equilibrium_distribution(rho, v1, v2, equations)
end

coords_min_lbm = (0.0, -1.0)
coords_max_lbm = (2.0, 1.0)
mesh_lbm = StructuredMesh(cells_per_dim_per_section,
                          coords_min_lbm, coords_max_lbm,
                          periodicity = (false, true))

# Supply equilibrium (Maxwellian) distribution function computed 
# from the Euler-variables as boundary values for the LBM equations
function coupling_function_Euler2LBM(x, u, equations_other, equations_own)
    u_prim_euler = cons2prim(u, equations_other)
    rho = u_prim_euler[1]
    v1 = u_prim_euler[2]
    v2 = u_prim_euler[3]

    return equilibrium_distribution(rho, v1, v2, equations_own)
end

boundary_conditions_lbm = (x_neg = BoundaryConditionCoupled(1, (:end, :i_forward),
                                                            Float64,
                                                            coupling_function_Euler2LBM),
                           x_pos = BoundaryConditionCoupled(1, (:begin, :i_forward),
                                                            Float64,
                                                            coupling_function_Euler2LBM),
                           y_neg = boundary_condition_periodic,
                           y_pos = boundary_condition_periodic)

semi_lbm = SemidiscretizationHyperbolic(mesh_lbm, eqs_lbm, initial_condition_lbm,
                                        solver_lbm;
                                        boundary_conditions = boundary_conditions_lbm)

# Create a semidiscretization that bundles the two semidiscretizations
semi = SemidiscretizationCoupled(semi_euler, semi_lbm)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback_euler = AnalysisCallback(semi_euler, interval = analysis_interval)
analysis_callback_lbm = AnalysisCallback(semi_lbm, interval = analysis_interval)
analysis_callback = AnalysisCallbackCoupled(semi,
                                            analysis_callback_euler,
                                            analysis_callback_lbm)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# Need to implement `cons2macroscopic` for `PolytropicEulerEquations2D`
# in order to be able to use this in the `SaveSolutionCallback` below
@inline function Trixi.cons2macroscopic(u, equations::PolytropicEulerEquations2D)
    u_prim = cons2prim(u, equations)
    p = pressure(u, equations)
    return SVector(u_prim[1], u_prim[2], u_prim[3], p)
end
function Trixi.varnames(::typeof(cons2macroscopic), ::PolytropicEulerEquations2D)
    ("rho", "v1", "v2", "p")
end

save_solution = SaveSolutionCallback(interval = 50,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2macroscopic)

cfl = 2.0
stepsize_callback = StepsizeCallback(cfl = cfl)

# Need special version of the LBM collision callback for a `SemidiscretizationCoupled`
@inline function Trixi.lbm_collision_callback(integrator)
    dt = get_proposed_dt(integrator)
    semi_coupled = integrator.p # Here `p` is the `SemidiscretizationCoupled`
    u_ode_full = integrator.u   # ODE Vector for the entire coupled system
    for (semi_index, semi_i) in enumerate(semi_coupled.semis)
        mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi_i)
        if equations isa LatticeBoltzmannEquations2D
            @unpack collision_op = equations

            u_ode_i = Trixi.get_system_u_ode(u_ode_full, semi_index, semi_coupled)
            u = Trixi.wrap_array(u_ode_i, mesh, equations, solver, cache)

            Trixi.@trixi_timeit Trixi.timer() "LBM collision" Trixi.apply_collision!(u, dt,
                                                                                     collision_op,
                                                                                     mesh,
                                                                                     equations,
                                                                                     solver,
                                                                                     cache)
        end
    end

    return nothing
end

collision_callback = LBMCollisionCallback()

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        collision_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK83();
            dt = 0.01, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
