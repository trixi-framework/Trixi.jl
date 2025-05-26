using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi
using LinearAlgebra: norm

###############################################################################
# Semidiscretizations of the Euler equations and Lattice-Boltzmann method using converter functions such that
# they are coupled across the domain boundaries to generate a periodic system.
#
# In this elixir, we have a square domain that is divided into a left and right half.
# On each half of the domain, an independent SemidiscretizationHyperbolic is created for each set of equations. 
# The two systems are coupled in the x-direction and are periodic in the y-direction.
# For a high-level overview, see also the figure below:
#
# (-2,  2)                                   ( 2,  2)
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
# (-2, -2)                                   ( 2, -2)

polydeg = 2
cells_per_dimension = (3, 6)

###########
# system #1
###########

gamma = 2.0 # Follows from LBM configuration
eqs_euler = CompressibleEulerEquations2D(gamma)

solver_euler = DGSEM(polydeg = 2, surface_flux = flux_lax_friedrichs)

# Follows from LBM configuration
function initial_condition_euler(x, t, equations::CompressibleEulerEquations2D)
    rho = pi
    v1 = 0.05773502691896255
    v2 = 0.05773502691896255
    p = 1.0471975511965979

    return prim2cons(SVector(rho, v1, v2, p), equations)
end

coords_min_euler = (-2.0, -2.0)
coords_max_euler = (0.0, 2.0)
mesh_euler = StructuredMesh(cells_per_dimension,
                            coords_min_euler, coords_max_euler,
                            periodicity = (false, true))

function coupling_function_LBM2Euler(x, u, equations_other, equations_own)
    return prim2cons(cons2macroscopic(u, equations_other), equations_own)
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
                                          solver_euler,
                                          boundary_conditions = boundary_conditions_euler)

###########
# system #2
###########

eqs_lbm = LatticeBoltzmannEquations2D(Ma = 0.1, Re = Inf)

# Quick & dirty implementation of the `flux_godunov` for cartesian, yet structured meshes.
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

initial_condition_lbm = initial_condition_constant
coords_min_lbm = (0.0, -2.0)
coords_max_lbm = (2.0, 2.0)
mesh_lbm = StructuredMesh(cells_per_dimension,
                          coords_min_lbm, coords_max_lbm,
                          periodicity = (false, true))

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
                                        solver_lbm,
                                        boundary_conditions = boundary_conditions_lbm)

# Create a semidiscretization that bundles all the semidiscretizations.
semi = SemidiscretizationCoupled(semi_euler, semi_lbm)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

analysis_callback_euler = AnalysisCallback(semi_euler, interval = 100)
analysis_callback_lbm = AnalysisCallback(semi_lbm, interval = 100)
analysis_callback = AnalysisCallbackCoupled(semi, analysis_callback_euler,
                                            analysis_callback_lbm)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 50,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

cfl = 1.0

stepsize_callback = StepsizeCallback(cfl = cfl)

# Need special version of the LBM collision callback
@inline function Trixi.lbm_collision_callback(integrator)
    dt = get_proposed_dt(integrator)
    semi_coupled = integrator.p
    u_ode_full = integrator.u
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

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 0.01, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
