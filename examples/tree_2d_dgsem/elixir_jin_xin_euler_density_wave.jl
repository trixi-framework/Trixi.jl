
using OrdinaryDiffEq
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the compressible Euler equations

epsilon_relaxation = 1.0e-6
a1 = a2 = a3 = a4 = 30.0
b1 = b2 = b3 = b4 = 30.0

equations_base = CompressibleEulerEquations2D(1.4)
velocities = (SVector(a1, a2, a3, a4), SVector(b1, b2, b3, b4))
equations = JinXinEquations(equations_base, epsilon_relaxation, velocities)

function initial_condition_density_wave(x, t, equations::CompressibleEulerEquations2D)
    v1 = 0.1
    v2 = 0.2
    rho = 1 + 0.98 * sinpi(2 * (x[1] + x[2] - t * (v1 + v2)))
    p = 20
    return prim2cons(SVector(rho, v1, v2, p),equations)
end

initial_condition = Trixi.InitialConditionJinXin(initial_condition_density_wave)
polydeg = 1
#basis = LobattoLegendreBasis(polydeg; polydeg_projection = 0)
basis = LobattoLegendreBasis(polydeg)

volume_integral = VolumeIntegralWeakForm()
solver = DGSEM(basis, Trixi.flux_upwind,VolumeIntegralWeakForm())
#solver = DGSEM(basis, Trixi.flux_upwind)

# solver = DGSEM(polydeg = 5, surface_flux = flux_central)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 30_000)

                semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)#,source_terms=source_terms_JinXin_Relaxation)


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

stepsize_callback = StepsizeCallback(cfl = 0.25)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        # save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
                                                     variables=(Trixi.density, pressure))
#sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
#sol = Trixi.solve(ode, Trixi.SimpleIMEX(), 
sol = solve(ode, SSPRK33(stage_limiter!),
dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
