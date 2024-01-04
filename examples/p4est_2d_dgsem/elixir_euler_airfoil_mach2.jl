
using Trixi
using OrdinaryDiffEq, Plots
using Downloads: download

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_mach2_flow(x, t, equations::CompressibleEulerEquations2D)
  # set the freestream flow parameters
  rho_freestream = 1.4
  v1 = 2.0
  v2 = 0.0
  p_freestream = 1.0

  prim = SVector(rho_freestream, v1, v2, p_freestream)
  return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach2_flow

# Supersonic inflow boundary condition.
# Calculate the boundary flux entirely from the external solution state, i.e., set
# external solution state values for everything entering the domain.
@inline function boundary_condition_supersonic_inflow(u_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t, surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
  u_boundary = initial_condition_mach2_flow(x, t, equations)
  flux = Trixi.flux(u_boundary, normal_direction, equations)

  return flux
end

# Supersonic outflow boundary condition.
# Calculate the boundary flux entirely from the internal solution state. Analogous to supersonic inflow
# except all the solution state values are set from the internal solution as everything leaves the domain
@inline function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                            surface_flux_function,
                                            equations::CompressibleEulerEquations2D)
  flux = Trixi.flux(u_inner, normal_direction, equations)

  return flux
end

d = 3

surface_flux = flux_lax_friedrichs # LOCAL Lax-Friedrichs = Rusanov flux
volume_flux  = flux_ranocha # For advanced DG technique (flux splitting)

basis = LobattoLegendreBasis(d)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

# DG Solver                                                 
solver = DGSEM(polydeg = d, surface_flux = surface_flux,
               volume_integral = volume_integral)

mesh_file = joinpath(@__DIR__, "NACA6412.inp")
isfile(mesh_file) ||
    download("https://gist.github.com/DanielDoehring/b434005800a1b0c0b4b50a8772283019/raw/1f6649b3370163d75d268176b51775f8685dd81d/NACA6412.inp",
            mesh_file)

boundary_symbols = [:PhysicalLine10, :PhysicalLine20, :PhysicalLine30, :PhysicalLine40]

mesh = P4estMesh{2}(mesh_file, polydeg = d, boundary_symbols=boundary_symbols)

boundary_conditions = Dict(:PhysicalLine10 => boundary_condition_supersonic_inflow,
                           :PhysicalLine20 => boundary_condition_outflow, # Right boundary
                           :PhysicalLine30 => boundary_condition_slip_wall, # Airfoil
                           :PhysicalLine40 => boundary_condition_outflow # Top/Bottom
                           )

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 4.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

# positivity limiter necessary for this example with strong shocks. Very sensitive
# to the order of the limiter variables, pressure must come first.
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-7, 1.0e-6),
                                                     variables = (pressure, Trixi.density))

# Run the simulation
###############################################################################
sol = solve(ode, #SSPRK104(; thread = OrdinaryDiffEq.True(), stage_limiter!);
            SSPRK104(; thread = OrdinaryDiffEq.True());
            dt = 42.0, # overwritten
            callback = callbacks);

summary_callback() # print the timer summary

plot(sol)

pd = PlotData2D(sol)
plot(pd["rho"])
plot!(getmesh(pd))