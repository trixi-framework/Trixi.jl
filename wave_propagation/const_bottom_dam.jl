using Plots
using LinearAlgebra
using OrdinaryDiffEq
using Trixi

equations = ShallowWaterEquations1D(gravity_constant=1.0, H0=5.0)

function initial_condition_wave(x, t, equations::ShallowWaterEquations1D)

  # Calculate primitive variables
  H = 5.0+ 3 *exp((-(5-x[1])^2))
  v = -10.0
  b = 1.0

  return prim2cons(SVector(H, v, b), equations)
end

Fr_vec = zeros(0)

function outflow_boundary(u_inner, orientation_or_normal, direction, x, t,
    surface_flux_function, equations::ShallowWaterEquations1D)

    g = equations.gravity
    h = u_inner[1]

    c = sqrt(g*h)

    v = velocity(u_inner, equations)

    Fr = abs(v)/c

    append!(Fr_vec, Fr)

    if Fr < 1
        # Subcritical outflow
        u_outer = SVector((equations.H0-1), 0, u_inner[3])
        if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
          flux = surface_flux_function(u_inner, u_outer, orientation_or_normal, equations)
        else # u_boundary is "left" of boundary, u_inner is "right" of boundary
          flux = surface_flux_function(u_outer, u_inner, orientation_or_normal, equations)
        end
    else
        # Supercritical outflow
        u_outer = SVector(u_inner[1], u_inner[2], u_inner[3])
        if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
          flux = surface_flux_function(u_inner, u_outer, orientation_or_normal, equations)
        else # u_boundary is "left" of boundary, u_inner is "right" of boundary
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

coordinates_min = 0.0
coordinates_max = 10.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=10_000,
                periodicity = false)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                     extra_analysis_integrals=(energy_kinetic,
                                                               energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

# Vector which sets the timesteps at which the solution will be saved
visnodes = range(tspan[1], tspan[2], length=300)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-8, reltol=1.0e-8,
            save_everystep=false, saveat=visnodes, # set saveat to visnodes to save at specified timesteps
            callback=callbacks);
summary_callback() # print the timer summary

nodes, _ = Trixi.gauss_lobatto_nodes_weights(4)

nodes = (nodes .+ 1) .* 0.625

pd = PlotData1D(sol)

nodess = zeros(32)

for i = 1:8
  for j = 1:4
    nodess[(i-1)*4+j] = pd.mesh_vertices_x[i] + nodes[j] 
  end
end

pyplot()

# display(maximum(Fr_vec))

animation = @animate for k= 1:300
    plot(nodess, sol.u[k][1:3:end]+sol.u[k][3:3:end], ylim=(0,10), label = "water surface")
    plot!(nodess, sol.u[k][3:3:end], label = "bottom topography", title = "t = $(sol.t[k])")
end

gif(animation, "const_bottom_dam.gif", fps=10)