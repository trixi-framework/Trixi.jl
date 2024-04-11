
using OrdinaryDiffEq
using Trixi
using Plots, LinearAlgebra


###############################################################################
# semidiscretization of the linear advection equation

equations = MaxwellEquation1D()

solver = DGSEM(polydeg = 3, surface_flux = flux_hll)

coordinates_min = 0.0
coordinates_max = 1.0

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000) # set maximum capacity of tree data structure

eps0 = 8.8541878128e-12
inv_eps0() = 1.0 / eps0

function source_terms_current(u, x, t, equations)
  s = - inv_eps0() * 0.0

  return SVector(s, 0.0)
end

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver, source_terms = source_terms_current)

J = jacobian_ad_forward(semi)

Eigenvalues = eigvals(J)

# Complex conjugate eigenvalues have same modulus
Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

# Sometimes due to numerical issues some eigenvalues have positive real part, which is erronous (for hyperbolic eqs)
Eigenvalues = Eigenvalues[real(Eigenvalues) .< 0]

EigValsReal = real(Eigenvalues)
EigValsImag = imag(Eigenvalues)

println(minimum(EigValsReal))
println(maximum(EigValsImag))

#plotdata = nothing
#plotdata = Plots.scatter(EigValsReal, EigValsImag, label = "Spectrum")
#display(plotdata)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1e-8));

summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()