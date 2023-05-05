using Plots
using Trixi
using Printf
using OrdinaryDiffEq

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-1,+1]^2
  slope = 15
  amplitude = 0.02
  B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * B
  v1 = 0.5 * (B - 1)
  v2 = 0.1 * sin(2 * pi * x[1])
  p = 1.0
  return prim2cons(SVector(rho, v1, v2, p), equations)
end

my_initial_condition = initial_condition_kelvin_helmholtz_instability

surface_flux = flux_lax_friedrichs
volume_flux  = flux_ranocha

polydeg = 3
inilevel = 2
maxlevel = 6

basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.002,
                                         alpha_min=0.0001,
                                         alpha_smooth=true,
                                         variable=Trixi.density)

# volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                  volume_flux_dg=volume_flux,
#                                                  volume_flux_fv=surface_flux)

volume_integral = VolumeIntegralWeakForm()

solver = DGSEM(basis, surface_flux, volume_integral)

# Warped rectangle that looks like a waving flag,
f1(s) = SVector(-1.0 + 0.1 * sin( pi * s), s)
f2(s) = SVector( 1.0 + 0.1 * sin( pi * s), s)
f3(s) = SVector(s, -1.0 + 0.1 * sin( pi * s))
f4(s) = SVector(s,  1.0 + 0.1 * sin( pi * s))

faces = (f1, f2, f3, f4)

Trixi.validate_faces(faces)
mapping_flag = Trixi.transfinite_mapping(faces)

if true

  # Simple periodic, n x n mesh.

  trees_per_dimension = (2, 2)
  mesh = T8codeMesh(trees_per_dimension,polydeg=polydeg, initial_refinement_level=inilevel, mapping=mapping_flag, periodicity=true)
  semi = SemidiscretizationHyperbolic(mesh, equations, my_initial_condition, solver)

else

  # Unstructured, crazy-looking mesh read in by a 'msh' file generated with 'gmsh'.

  # It sometimes Trixi crashes for meshes loaded from 'gmesh' files because they can have
  # flipped domains, e.g. [-1, 1] x [ 1,-1]. This is the case for the loaded mesh here.
  # The following linear transformation flips the domain back. Cool, eh?!
  coordinates_min = (-1.0,  1.0)
  coordinates_max = ( 1.0, -1.0)

  mapping_flip = Trixi.coordinates2mapping(coordinates_min, coordinates_max)

  my_mapping(x,y) = mapping_flag(mapping_flip(x,y)...)

  mesh_file = joinpath(@__DIR__,"meshfiles/unstructured_quadrangle.msh")

  boundary_conditions = Dict(
    :all => BoundaryConditionDirichlet(my_initial_condition),
  )

  mesh = T8codeMesh{2}(mesh_file, polydeg=polydeg,
                      mapping=my_mapping,
                      initial_refinement_level=inilevel)

  semi = SemidiscretizationHyperbolic(mesh, equations, my_initial_condition, solver,
                                      boundary_conditions=boundary_conditions)

end

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

# Not supported yet.
# save_solution = SaveSolutionCallback(interval=100,
#                                      save_initial_solution=true,
#                                      save_final_solution=true,
#                                      solution_variables=cons2prim)

amr_indicator = IndicatorLöhner(semi, variable=Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=0,
                                      med_level=0, med_threshold=0.05,
                                      max_level=maxlevel, max_threshold=0.1)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=1,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=0.8)

function my_save_plot(plot_data, variable_names;
                   show_mesh=true, plot_arguments=Dict{Symbol,Any}(),
                   time=nothing, timestep=nothing)

  title = @sprintf("2D KHI | Trixi.jl | 4th-order DG | AMR w/ t8code: t = %3.2f", time)

  sol = plot_data["rho"]

  Plots.plot(sol,
    clim=(0.25,2.25),
    colorbar_title="\ndensity",
    title=title,titlefontsize=10,
    dpi=300,
  )
  Plots.plot!(getmesh(plot_data),linewidth=0.5)

  mkpath("out")
  filename = joinpath("out", @sprintf("solution_%06d.png", timestep))
  Plots.savefig(filename)
end

visualization_callback = VisualizationCallback(plot_creator=my_save_plot,interval=50, clims=(0,1.1), show_mesh=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        # save_solution,
                        amr_callback,
                        visualization_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
