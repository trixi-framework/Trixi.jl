#src # Adaptive mesh refinement

# Adaptive mesh refinement (AMR) is a method of adapting the accuracy of a solution within certain
# sensitive or turbulent regions of simulation. In those critical regions of the domain the
# simulation uses more elements than in others. This is adapted dynamically during the time of
# simulation.


# # Implementation in Trixi
# In [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) AMR is possible for the mesh types
# [`TreeMesh`](@ref) and [`P4estMesh`](@ref). Both meshes are organized in a tree structure
# and therefore, each element can be refined independently.

# ### Indicators
# Indicators decide which regions of the domain are refined. In Trixi, you can use for instance
# [`IndicatorLöhner`](@ref) and [`IndicatorHennemannGassner`](@ref). `IndicatorLöhner` (also callable with
# `IndicatorLoehner`) is based on a FEM indicator by
# [Löhner (1987)](https://doi.org/10.1016/0045-7825(87)90098-3) and estimates a weighted second
# derivative of a specified variable locally.
# ````julia
# amr_indicator = IndicatorLöhner(semi, variable=variable)
# ````
# `IndicatorHennemannGassner`, also used as an shock-capturing indicator, was developed by
# [Hennemann et al. (2021)](https://doi.org/10.1016/j.jcp.2020.109935) and is explained in detail
# in the [tutorial about shock-capturing](@ref shock_capturing). It can be implemented in the
# following way.
# ````julia
# amr_indicator = IndicatorHennemannGassner(semi,
#                                           alpha_max=0.5,
#                                           alpha_min=0.001,
#                                           alpha_smooth=true,
#                                           variable=variable)
# ````

# Another indicator is the very basic `IndicatorMax`. It indicates the maximal value of a variable
# and is therefore a mainly theoretical application. But it might be useful for the basic understanding
# of the implementation of indicators and AMR in Trixi.
# ````julia
# amr_indicator = IndicatorMax(semi, variable=variable)
# ````


# ### Controllers
# The spatial discretization into elements is tree based for both AMR supporting mesh types `TreeMesh`
# and `P4estMesh`, the higher the level in the tree the higher the level of refinement.

# To transfer specific indicator values to a specific level of refinement, Trixi uses controllers.
# They are build in three levels, which means there is a base level of refinement `base_level`.
# Then, there is a medium level `med_level` for indicator values above the threshold `med_threshold`
# and equally, a maximal level `max_level` for values above `max_threshold`.
# This variant of controller is called [`ControllerThreeLevel`](@ref) in Trixi.
# ````julia
# amr_controller = ControllerThreeLevel(semi, amr_indicator;
#                                       base_level=1,
#                                       med_level=base_level, med_threshold=0.0,
#                                       max_level=base_level, max_threshold=1.0)
# ````

# An extension is [`ControllerThreeLevelCombined`](@ref), which uses two different indicators.
# The primary indicator works in the same way like the sole indicator for `ControllerThreeLevel`.
# The second indicator with own maximum threshold adds the property, that the target level is set to
# `max_level` if this indicator's value is greater than `max_threshold_secondary`.
# ````julia
# amr_controller = ControllerThreeLevelCombined(semi, indicator_primary, indicator_secondary;
#                                               base_level=1,
#                                               med_level=base_level, med_threshold=0.0,
#                                               max_level=base_level, max_threshold=1.0,
#                                               max_threshold_secondary=1.0)
# ````


# ### Callback
# The defined indicator and controller are added to the simulation through the callback [`AMRCallback`](@ref).
# It contains a semidiscretization `semi`, the controller `amr_controller` and the parameters `interval`,
# `adapt_initial_condition` and `adapt_initial_condition_only_refine`.

# Adaptive mesh refinement will be performed every `interval` time steps. `adapt_initial_condition` says
# that the initial condition already should be adapted before the first time step. And with
# `adapt_initial_condition_only_refine=true` the mesh is only refined at the beginning but not coarsened.
# ````julia
# amr_callback = AMRCallback(semi, amr_controller,
#                            interval=5,
#                            adapt_initial_condition=true,
#                            adapt_initial_condition_only_refine=true)
# ````


# # Exemplary simulation
# Here, we implement a simple AMR simulation of the 2D linear advection equation for a Gaussian pulse.
# Some more complicated AMR simulations can be found on Trixi's youtube channel
# ["Trixi Framework"](https://www.youtube.com/channel/UCpd92vU2HjjTPup-AIN0pkg), for instance an
# [ideal MHD rotor on unstructured AMR mesh](https://www.youtube.com/watch?v=Iei7e9oQ0hs)
# and a [purely hyperbolic simulation of a Sedov blast wave with self-gravity](https://www.youtube.com/watch?v=dxgzgteJdOA).

using OrdinaryDiffEq
using Trixi

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_gauss
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-5.0, -5.0)
coordinates_max = ( 5.0,  5.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=30_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan);

# We use the simple `IndicatorMax` as our AMR indicator. Therefore, regions with a high maximum
# are refined. This is no really useful numerical application, but a nice demonstration example.
amr_indicator = IndicatorMax(semi, variable=first)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=4,
                                      med_level=5, med_threshold=0.1,
                                      max_level=6, max_threshold=0.6)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=1.6)

callbacks = CallbackSet(amr_callback, stepsize_callback);

# Running the simulation.
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# We plot the solution and add the refined mesh at the end of the simulation.
using Plots
pd = PlotData2D(sol)
plot(pd)
plot!(getmesh(pd))