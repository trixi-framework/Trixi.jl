###############################################################################
# This example is equivalent to tree_1d_dgsem/elixir_shallowwater_wall.jl,    #            
# but instead of a function for the bottom topography, this version uses a    #
# cubic B-spline interpolation with not-a-knot boundary condition to          #
# approximate the bottom topography. Interpolation points are provided        #
# via a gist.                                                                 #
###############################################################################

using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations

equations = ShallowWaterEquations1D(gravity_constant=9.812, H0=2.0)
  
###############################################################################
# The data for the bottom topography is saved as a .txt-file in a gist.
# To create the data, the following function
# bottom_topography(x) = (1.5 / exp( 0.5 * ((x - 1.0)^2))
#                         + 0.75 / exp(0.5 * ((x + 1.0)^2)))
# has been evaluated at 11 equally spaced points between [-5,5] and the
# resulting values have been saved.
spline_data    = download("https://gist.githubusercontent.com/maxbertrand1996/221adf82516b747457440c0217c1cc57/raw/82b2ac03d29613d5afc6cf3b17a5820dc38021ea/data_swe_wall_1D")

# Spline Interpolation
spline         = cubic_b_spline(spline_data; boundary = "not-a-knot")
spline_func(x) = spline_interpolation(spline, x)

function initial_condition_stone_throw(x, t, equations::ShallowWaterEquations1D)
    # Set up polar coordinates
    inicenter = SVector(0.15)
    x_norm = x[1] - inicenter[1]
    r = abs(x_norm)
  
    # Calculate primitive variables
    H = equations.H0
    v = r < 0.6 ? 1.75 : 0.0
    
    b = spline_func(x[1])
    
    return prim2cons(SVector(H, v, b), equations)
end
  
initial_condition = initial_condition_stone_throw

boundary_condition = boundary_condition_slip_wall

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=3, surface_flux=(flux_hll, flux_nonconservative_fjordholm_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = -5.0
coordinates_max = 5.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=10_000,
                periodicity=false)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                     extra_analysis_integrals=(energy_kinetic,
                                                               energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

# Vector which sets the timesteps at which the solution will be saved
visnodes = range(tspan[1], tspan[2], length=300)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-8, reltol=1.0e-8,
            save_everystep=false, saveat=visnodes, # set saveat to visnodes to save at specified timesteps
            callback=callbacks);
summary_callback() # print the timer summary

# Gif code inspired by 
# https://trixi-framework.github.io/Trixi.jl/stable/tutorials/non_periodic_boundaries/
using Plots
@gif for step in 1:length(sol.u)
    plot(sol.u[step], semi, ylim=(0, 3), legend=false, title="time t=$(round(sol.t[step], digits=5))")
end