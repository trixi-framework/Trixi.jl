
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

# Modify the manufactured solution test to use `L = sqrt(2)` in the initial and source terms
# such that testing works on the flipped mesh
function initial_condition_convergence_upwind(x, t, equations::CompressibleEulerEquations2D)
  c = 2
  A = 0.1
  L = sqrt(2)
  f = 1/L
  ω = 2 * pi * f
  ini = c + A * sin(ω * (x[1] + x[2] - t))

  rho = ini
  rho_v1 = ini
  rho_v2 = ini
  rho_e = ini^2

  return SVector(rho, rho_v1, rho_v2, rho_e)
end

@inline function source_terms_convergence_upwind(u, x, t, equations::CompressibleEulerEquations2D)
  # Same settings as in `initial_condition`
  c = 2
  A = 0.1
  L = sqrt(2)
  f = 1/L
  ω = 2 * pi * f
  γ = equations.gamma

  x1, x2 = x
  si, co = sincos(ω * (x1 + x2 - t))
  rho = c + A * si
  rho_x = ω * A * co
  # Note that d/dt rho = -d/dx rho = -d/dy rho.

  tmp = (2 * rho - 1) * (γ - 1)

  du1 = rho_x
  du2 = rho_x * (1 + tmp)
  du3 = du2
  du4 = 2 * rho_x * (rho + tmp)

  return SVector(du1, du2, du3, du4)
end

initial_condition = initial_condition_convergence_upwind

###############################################################################
# Get the FDSBP approximation operator

D_upw = upwind_operators(SummationByPartsOperators.Mattsson2017,
                         derivative_order=1,
                         accuracy_order=4,
                         xmin=-1.0, xmax=1.0,
                         N=10)

flux_splitting = splitting_lax_friedrichs
solver = FDSBP(D_upw,
               surface_integral=SurfaceIntegralStrongForm(FluxUpwind(flux_splitting)),
               volume_integral=VolumeIntegralUpwind(flux_splitting))

###############################################################################
# Get the curved quad mesh from a file (downloads the file if not available locally)
default_mesh_file = joinpath(@__DIR__, "mesh_multiple_flips.mesh")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/b434e724e3972a9c4ee48d58c80cdcdb/raw/9a967f066bc5bf081e77ef2519b3918e40a964ed/mesh_multiple_flips.mesh",
                                       default_mesh_file)

mesh_file = default_mesh_file

mesh = UnstructuredMesh2D(mesh_file, periodicity=true)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_convergence_upwind)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback,
                        alive_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(), abstol=1.0e-9, reltol=1.0e-9,
            save_everystep=false, callback=callbacks)
summary_callback() # print the timer summary
