using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using Revise
using Trixi
using LinearAlgebra
using Trixi: StartUpDG
###############################################################################
# semidiscretization of the compressible Navier-Stokes equations
prandtl_number() = 0.72
mu() = 6.25e-4 # equivalent to Re = 1600
equations = CompressibleEulerEquations3D(1.4)
equations_parabolic_limiting = CompressibleNavierStokesDiffusion3D(equations, mu = mu(),
                                                                   Prandtl = prandtl_number())
equations_parabolic_av = CompressibleNavierStokesDiffusion3D(equations, mu = mu(),
                                                             Prandtl = prandtl_number(),
                                                             gradient_variables = GradientVariablesEntropy());
function initial_condition_taylor_green_vortex(x, t,
                                               equations::CompressibleEulerEquations3D)
    A = 1.0 # magnitude of speed
    Ms = 0.1 # maximum Mach number
    rho = 1.0
    v1 = A * sin(x[1]) * cos(x[2]) * cos(x[3])
    v2 = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
    v3 = 0.0
    p = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
    p = p +
        1.0 / 16.0 * A^2 * rho *
        (cos(2 * x[1]) * cos(2 * x[3]) + 2 * cos(2 * x[2]) + 2 * cos(2 * x[1]) +
         cos(2 * x[2]) * cos(2 * x[3]))
    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_taylor_green_vortex
coordinates_min = (-1.0, -1.0, -1.0) .* pi
coordinates_max = (1.0, 1.0, 1.0) .* pi
degree = 3
initial_refine = 4
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refine,
                n_cells_max = 400_000, periodicity = true)
volume_flux = flux_kennedy_gruber
surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(degree)
indicator_ec = IndicatorEntropyCorrection(equations, basis)
volume_integral_default = VolumeIntegralWeakForm()
volume_integral_default = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral_entropy_stable = VolumeIntegralPureLGLFiniteVolume(surface_flux)
volume_integral = VolumeIntegralAdaptive(indicator_ec,
                                         volume_integral_default,
                                         volume_integral_entropy_stable)
#volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(polydeg = degree, surface_flux = flux_lax_friedrichs,
               volume_integral = volume_integral)
solver_parabolic = ParabolicFormulationBassiRebay1()
#solver_parabolic = ParabolicFormulationLocalDG()
semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic_limiting),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))
volume_flux = flux_central
solver = DGSEM(polydeg = degree, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
N = Trixi.polydeg(solver)
filter = [((i - 1) / N)^2 for i in 1:(N + 1)]
nodes = Trixi.get_nodes(solver.basis)
# construct generalized vandermonde matrix (line), and use tensor product
# to create 3D vandermonde matrix for cube style element
#rd = RefElemData(Line(), SBP(), N)
V = StartUpDG.NodesAndModes.vandermonde(Line(), N, nodes)
#VDM = Matrix{Float64}(V ⊗ V ⊗ V)
VDM = kron(V, V, V)
semi_av = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic_av),
                                                initial_condition, solver;
                                                VDM = VDM, filter = filter,
                                                combine_rhs = Trixi.True(),
                                                solver_parabolic = solver_parabolic,
                                                boundary_conditions = (boundary_condition_periodic,
                                                                       boundary_condition_periodic));
###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 0.1)
ode = semidiscretize(semi_av, tspan)
summary_callback = SummaryCallback();
save_callback = SaveSolutionCallback(dt = 0.1,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:enstrophy,));
analysis_interval = 50
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     output_directory = "out",
                                     #analysis_filename="test.dat",
                                     analysis_filename = "TGV16.dat",
                                     extra_analysis_integrals = (energy_kinetic,
                                                                 energy_internal,
                                                                 enstrophy));
alive_callback = AliveCallback(analysis_interval = analysis_interval);
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback);
###############################################################################
# run the simulation
# time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol = 1e-6, reltol = 1e-4,
            saveat = 4.0, ode_default_options()..., callback = callbacks)
# sol = solve(ode, Tsit5(); abstol = 1e-6, reltol = 1e-4,
#              saveat=4.0, ode_default_options()..., callback = callbacks)
