using Plots
using LinearAlgebra
using OrdinaryDiffEq
using Trixi
using Downloads: download

# Downlowad data from gist
Rhine_data = download("https://gist.githubusercontent.com/maxbertrand1996/87f57c6d1c3c99e1ace67f52d65a80f4/raw/fd29b9bda097532ce3d704cfe224c7d7f4efeab1/Rhine_data_2D_10.txt")

file = open(Rhine_data)
lines = readlines(file)
close(file)

n = parse(Int64, lines[2])
m = parse(Int64, lines[4])

x_1d  = [parse(Float64, val) for val in lines[6:(5+n)]]
y     = [parse(Float64, val) for val in lines[(7+n):(6+n+m)]]
z_tmp = [parse(Float64, val) for val in lines[(8+n+m):end]]

z = Matrix(transpose(reshape(z_tmp, (n, m))))

y_1d = z[50,:]

# Define B-spline structure
spline_struct = cubic_b_spline(x_1d,y_1d)
# Define B-spline interpolation function
spline_func(x) = spline_interpolation(spline_struct, x)

equations = ShallowWaterEquations1D(gravity_constant=1.0, H0=60.0)

function initial_condition_wave(x, t, equations::ShallowWaterEquations1D)

  inicenter = SVector(357500.0)
  x_norm = x[1] - inicenter[1]
  r = abs(x_norm)

  # Calculate primitive variables
  H = r < 100 ? 65.0 : equations.H0
  v = 5.0#r < 100 ? 5.0 : 0.0
  b = 50.0#spline_func(x[1])

  return prim2cons(SVector(H, v, b), equations)
end

function outflow_boundary(u_inner, orientation_or_normal, direction, x, t,
  surface_flux_function, equations::ShallowWaterEquations1D)

  g = equations.gravity
  h = u_inner[1]

  c = sqrt(g*h)

  v = velocity(u_inner, equations)

  Fr = abs(v)/c

  # display(Fr)

  if Fr < 1
    # Subcritical outflow
    u_outer = SVector(equations.H0, u_inner[2], u_inner[3])
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    #   u_outer = SVector(equations.H0, u_inner[2], u_inner[3])
      flux = surface_flux_function(u_inner, u_outer, orientation_or_normal, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    #   u_outer = SVector(equations.H0, -u_inner[2], u_inner[3])
      flux = surface_flux_function(u_outer, u_inner, orientation_or_normal, equations)
    end
  else
    # Supercritical outflow
    u_outer = SVector(u_inner[1], u_inner[2], u_inner[3])
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    #   u_outer = SVector(u_inner[1], u_inner[2], u_inner[3])
      flux = surface_flux_function(u_inner, u_outer, orientation_or_normal, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    #   u_outer = SVector(u_inner[1], -u_inner[2], u_inner[3])
      flux = surface_flux_function(u_outer, u_inner, orientation_or_normal, equations)
    end
  end
        
end

initial_condition = initial_condition_wave

boundary_condition = outflow_boundary

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=3, surface_flux=(flux_hll, flux_nonconservative_fjordholm_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = x_1d[1]
coordinates_max = x_1d[end]
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

# save_solution = SaveSolutionCallback(interval=10,
#                                      save_initial_solution=true,
#                                      save_final_solution=true,
#                                      output_directory="initial_condition_wave")

# vis = VisualizationCallback(interval=10,
#                                      solution_variables=cons2prim,
#                                      variable_names=["H"],
#                                      show_mesh=false,
#                                      plot_data_creator=PlotData1D,
#                                      plot_creator=show_plot)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)#, save_solution)#, vis)

# Vector which sets the timesteps at which the solution will be saved
visnodes = range(tspan[1], tspan[2], length=300)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-8, reltol=1.0e-8,
            save_everystep=false, saveat=visnodes, # set saveat to visnodes to save at specified timesteps
            callback=callbacks);
summary_callback() # print the timer summary

pd = PlotData1D(sol)

pyplot()
plot(pd["H"])
plot!(pd["b"])

# nodes, _ = Trixi.gauss_lobatto_nodes_weights(4)

# nodes = (nodes .+ 1) .* 0.375

# pd = PlotData1D(sol)

# nodess = zeros(32)

# for i = 1:8
#   for j = 1:4
#     nodess[(i-1)*4+j] = pd.mesh_vertices_x[i] + nodes[j] 
#   end
# end

# pyplot()

# animation = @animate for k= 1:300
#     plot(nodess, sol.u[k][1:3:end]+sol.u[k][3:3:end], ylim=(40,80), label = "water surface")
#     plot!(nodess, sol.u[k][3:3:end], label = "bottom topography", title = "t = $(sol.t[k])")
# end

# gif(animation, "Wave_propagation.gif", fps=10)