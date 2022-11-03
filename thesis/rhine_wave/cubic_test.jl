# Include packages
using OrdinaryDiffEq
using Trixi
using Plots

x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
y = [3.0, 1.0, 2.0, 0.0, 2.0, 3.0, 2.0]

# Define B-spline structure
spline_struct = cubic_b_spline(x,y)
# Define B-spline interpolation function
spline_func(x) = spline_interpolation(spline_struct, x)

equations = ShallowWaterEquations1D(gravity_constant=1.0, H0=5.0)

function initial_condition_wave(x, t, equations::ShallowWaterEquations1D)

  inicenter = SVector(3.0)
  x_norm = x[1] - inicenter[1]
  r = abs(x_norm)

  # Calculate primitive variables
  H = r < 0.5 ? 7.0 : equations.H0
  v = 0.0
  b = spline_func(x[1])

  return prim2cons(SVector(H, v, b), equations)
end

function boundary_outflow(u_inner, orientation_or_normal, direction, x, t,
  surface_flux_function, equations::ShallowWaterEquations1D)

  g = equations.gravity
  h = u_inner[1]

  c = sqrt(g*h)

  v = velocity(u_inner, equations)

  Fr = abs(v)/c

  if Fr > 1
    # Supercritical outflow
    u_outer = SVector(u_inner[1], u_inner[2], u_inner[3])
    # calculate the boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
      flux = surface_flux_function(u_inner, u_outer, orientation_or_normal, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
      flux = surface_flux_function(u_outer, u_inner, orientation_or_normal, equations)
    end

    return flux

  else
    # Subcritical outflow
    # Impulse from inside, height and bottom from outside
    u_outer = SVector(equations.H0, u_inner[2], u_inner[3])
    # calculate the boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
      flux = surface_flux_function(u_inner, u_outer, orientation_or_normal, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
      flux = surface_flux_function(u_outer, u_inner, orientation_or_normal, equations)
    end

    return flux
  end
end

initial_condition = initial_condition_wave

boundary_condition = boundary_outflow
###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=3, surface_flux=(flux_hll, flux_nonconservative_fjordholm_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = x[1]
coordinates_max = x[end]
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=10_000)#,
                # periodicity = false)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)#,
                                    # boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                     extra_analysis_integrals=(energy_kinetic,
                                                               energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     output_directory="initial_condition_wave")

# vis = VisualizationCallback(interval=10,
#                                      solution_variables=cons2prim,
#                                      variable_names=["H"],
#                                      show_mesh=false,
#                                      plot_data_creator=PlotData1D,
#                                      plot_creator=show_plot)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)#, vis)

# Vector which sets the timesteps at which the solution will be saved
visnodes = range(tspan[1], tspan[2], length=30)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-8, reltol=1.0e-8,
            save_everystep=false, saveat=visnodes, # set saveat to visnodes to save at specified timesteps
            callback=callbacks);
summary_callback() # print the timer summary

nodes, _ = Trixi.gauss_lobatto_nodes_weights(4)

nodes = (nodes .+ 1) .* 0.375

pd = PlotData1D(sol)

nodess = zeros(32)

for i = 1:8
  for j = 1:4
    nodess[(i-1)*4+j] = pd.mesh_vertices_x[i] + nodes[j] 
  end
end

pyplot()

animation = @animate for k= 1:30
    plot(nodess, sol.u[k][1:3:end]+sol.u[k][3:3:end], ylim=(0,10), label = "water surface")
    plot!(nodess, sol.u[k][3:3:end], label = "bottom topography", title = "t = $(sol.t[k])")
end

gif(animation, "Wave_propagation.gif", fps=10)