using OrdinaryDiffEqLowStorageRK
using Trixi
using LinearAlgebra: norm

###############################################################################
# semidiscretization of the Lattice-Boltzmann equations for the D2Q9 scheme

equations = LatticeBoltzmannEquations2D(Ma = 0.1, Re = Inf)

initial_condition = initial_condition_constant

solver = DGSEM(polydeg = 3, surface_flux = flux_godunov)

cells_per_dimension = (16, 16)
coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = StructuredMesh(cells_per_dimension,
                      coordinates_min, coordinates_max)

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

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2macroscopic)

stepsize_callback = StepsizeCallback(cfl = 1.0)

collision_callback = LBMCollisionCallback()

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        collision_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
