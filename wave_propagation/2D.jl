
using Plots
using LinearAlgebra
using OrdinaryDiffEq
using Trixi

Rhine_data = download("https://gist.githubusercontent.com/maxbertrand1996/a30db4dc9f5427c78160321d75a08166/raw/fa53ceb39ac82a6966cbb14e1220656cf7f97c1b/Rhine_data_2D_40.txt")

spline_struct = bicubic_b_spline(Rhine_data)
spline_func(x,y) = spline_interpolation(spline_struct, x, y)

equations = ShallowWaterEquations2D(gravity_constant=9.81, H0=55.0)

function initial_condition_wave(x, t, equations::ShallowWaterEquations2D)

  inicenter = SVector(357490.0, 5646519.0)
  x_norm = x - inicenter
  r = norm(x_norm)

  # Calculate primitive variables
  H =  r < 50 ? 65.0 : 55.0
  v1 = 0.0
  v2 = 0.0

  x1, x2 = x
  b = spline_func(x1, x2)

  return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_wave

boundary_condition = boundary_condition_slip_wall

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=3, surface_flux=(flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = (spline_struct.x[1], spline_struct.y[1])
coordinates_max = (spline_struct.x[end], spline_struct.y[end])
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=10_000,
                periodicity = false)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 100.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                     extra_analysis_integrals=(energy_kinetic,
                                                               energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

# save_solution = SaveSolutionCallback(interval=100,
#                                      save_initial_solution=true,
#                                      save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)#, save_solution)

# Vector which sets the timesteps at which the solution will be saved
visnodes = range(tspan[1], tspan[2], length=300)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-8, reltol=1.0e-8,
            save_everystep=false, #saveat=visnodes, 
            callback=callbacks);
summary_callback() # print the timer summary

pd = PlotData2D(sol)

pyplot()
# surface(pd["b"])
# wireframe!(pd["H"])
surface(pd.x[2:end-1], pd.y[2:end-1], pd.data[4][2:end-1, 2:end-1])
wireframe!(pd.x[2:end-1], pd.y[2:end-1], pd.data[1][2:end-1, 2:end-1])

# function fill_sol_mat(f, x, y)
    
#     # Get dimensions for solution matrix
#     n = length(x)
#     m = length(y)
  
#     # Create empty solution matrix
#     z = zeros(n,m)
  
#     # Fill solution matrix
#     for i in 1:n, j in 1:m
#       # Evaluate spline functions
#       # at given x,y values
#       z[j,i] = f(x[i], y[j])
#     end
  
#   # Return solution matrix
#   return z
# end

# x_int = Vector(LinRange(spline_struct.x[1], spline_struct.x[end], 100))
# y_int = Vector(LinRange(spline_struct.y[1], spline_struct.y[end], 100))
# z_int = fill_sol_mat(spline_func, x_int, y_int)

# surface(x_int, y_int, z_int, title = "Own representation")
# # surface(pd["b"], title ="PlotData2D")