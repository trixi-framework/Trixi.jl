
using OrdinaryDiffEq
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

function initial_condition_density_wave2(x, t, equations::CompressibleEulerEquations2D)
    v1 = 0.1
    v2 = 0.2
    rho = 1 + 0.98 * sinpi(2 * (x[1] + x[2] - t * (v1 + v2)))
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    p = 20
    rho_e = p / (equations.gamma - 1) + 1 / 2 * rho * (v1^2 + v2^2)
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
initial_condition = initial_condition_density_wave2

surface_flux = flux_central
polydeg = 17
basis = LobattoLegendreBasis(polydeg; polydeg_projection = 2 * polydeg, polydeg_cutoff = 5)
volume_integral = VolumeIntegralWeakForm()
solver = DGSEM(basis, surface_flux, volume_integral)

# solver = DGSEM(polydeg = 5, surface_flux = flux_central)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 30_000)

# @info "Create semi..."
# semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

# @info "Compute Jacobian..."
# J = jacobian_ad_forward(semi)

# @info "Compute eigenvalues..."
# λ = eigvals(J)

# @info "max real part" maximum(real.(λ))
# # @info "Plot spectrum..."
# # scatter(real.(λ), imag.(λ), label="central flux")
# wololo

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100 * 20
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.6)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        # save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
