
using OrdinaryDiffEq
using Trixi

###############################################################################
#  setup the equations

γ = 1.4
equations = CompressibleEulerEquations1D(γ)

###############################################################################
#  setup the GSBP DG discretization that uses the Gauss operators from Chan et al.

surface_flux = FluxLaxFriedrichs()
volume_flux = flux_ranocha
dg = DGMulti(polydeg = 3,
             element_type = Line(),
             approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
#  setup the test problem (no source term needed for linear advection)

initial_condition = initial_condition_shu_osher

###############################################################################
#  setup the boundary condition 

left_boundary(x, tol=50*eps())  = abs(x[1]+5.0)<tol
right_boundary(x, tol=50*eps()) = abs(x[1]-5.0)<tol
is_on_boundary = Dict(:left => left_boundary, :right => right_boundary)

boundary_condition_shu_osher = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (;  :left => boundary_condition_shu_osher,
                         :right => boundary_condition_shu_osher)

###############################################################################
#  setup the 1D mesh

mesh = DGMultiMesh(dg,
                   cells_per_dimension=(128,),
                   coordinates_min=(-5.0,),
                   coordinates_max=( 5.0,),
                   periodicity=false,
                   is_on_boundary=is_on_boundary)

###############################################################################
#  setup the semidiscretization and ODE problem

semi = SemidiscretizationHyperbolic(mesh,
                                    equations,
                                    initial_condition,
                                    dg,
                                    boundary_conditions=boundary_conditions)

tspan = (0.0, 1.8)
ode = semidiscretize(semi, tspan)

###############################################################################
#  setup the callbacks

# prints a summary of the simulation setup and resets the timers
summary_callback = SummaryCallback()

# analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100, uEltype=real(dg))

# handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.125)

# collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(),
            dt=1e-6, save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

using Plots 

plot(sol)