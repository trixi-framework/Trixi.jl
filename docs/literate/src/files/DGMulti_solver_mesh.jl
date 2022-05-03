#src # `DGMulti` solver and mesh

# The basic idea and implementation of the [`DGMulti`](@ref) solver is already explained in
# section ["Meshes"](@ref DGMulti).
# Here, we want to give some examples and a quick overview about the options with `DGMulti`.

# We start with a simple example we already used in the [tutorial about flux differencing](@ref fluxDiffExample).
# There, we implemented a simulation with [`initial_condition_weak_blast_wave`](@ref) for the
# 2D compressible Euler equations [`CompressibleEulerEquations2D`](@ref) and used the DG formulation
# with flux differencing using volume flux [`flux_ranocha`](@ref) and surface flux [`flux_lax_friedrichs`](@ref).

# Here, we want to implement the equivalent example, only now using the `DGMulti` solver
# instead of [`DGSEM`](@ref).

using Trixi, OrdinaryDiffEq

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_weak_blast_wave

# To use the Gauss-Lobatto nodes again, we choose `approximation_type=SBP()` in the solver.
# Since we want to start with a Cartesian domain discretized with squares, we use the element
# type `Quad()`.
dg = DGMulti(polydeg = 3,
             element_type = Quad(),
             approximation_type = SBP(),
             surface_flux = flux_lax_friedrichs,
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

mesh = DGMultiMesh(dg,
                   cells_per_dimension=(32, 32), # initial_refinement_level = 5
                   coordinates_min=(-2.0, -2.0),
                   coordinates_max=( 2.0,  2.0),
                   periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    boundary_conditions=boundary_condition_periodic)
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=10)
analysis_callback = AnalysisCallback(semi, interval=100, uEltype=real(dg))
callbacks = CallbackSet(analysis_callback, alive_callback);

# Run the simulation with the same time integration algorithm as before.
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-6, reltol=1.0e-6,
            callback=callbacks, save_everystep=false);
#-
using Plots
pd = PlotData2D(sol)
plot(pd)
#-
plot(pd["rho"])
plot!(getmesh(pd))

# This simulation is not as fast as the equivalent with `TreeMesh` since the functions are not
# implemented that efficiently. Figure 4 in ["Efficient implementation of modern entropy stable
# and kinetic energy preserving discontinuous Galerkin methods for conservation laws"](https://arxiv.org/abs/2112.10517)
# (2021) provides a nice runtime comparison between the different mesh types. On the other hand,
# the functions are more general and thus we have more option we can choose from.


# ## Simulation with Gauss nodes
# For instance, we can change the approximation type of our simulation.
using Trixi, OrdinaryDiffEq
equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_weak_blast_wave

# We now use Gauss nodes instead of Gauss-Lobatto nodes which can be done for the element types
# `Quad()` and `Hex()`. Therefore, we set `approximation_type=GaussSBP()`. Alternatively, we
# can use a modal approach using the approximation type `Polynomial()`.
dg = DGMulti(polydeg = 3,
             element_type = Quad(),
             approximation_type = GaussSBP(),
             surface_flux = flux_lax_friedrichs,
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

mesh = DGMultiMesh(dg,
             cells_per_dimension=(32, 32), # initial_refinement_level = 5
             coordinates_min=(-2.0, -2.0),
             coordinates_max=( 2.0,  2.0),
             periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                              boundary_conditions=boundary_condition_periodic)
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=10)
analysis_callback = AnalysisCallback(semi, interval=100, uEltype=real(dg))
callbacks = CallbackSet(analysis_callback, alive_callback);

sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-6, reltol=1.0e-6,
            callback=callbacks, save_everystep=false);
#-
using Plots
pd = PlotData2D(sol)
plot(pd)


# ## Simulation with triangular elements
# Also, we can set another element type. We want to use triangles now.
using Trixi, OrdinaryDiffEq
equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_weak_blast_wave

# Since there is no direct equivalent to Gauss-Lobatto nodes on triangles, the approximation type
# `SBP()` now uses Gauss-Lobatto nodes on faces. More details can be found in the documentation of
# [StartUpDG.jl](https://jlchan.github.io/StartUpDG.jl/dev/RefElemData/#RefElemData-based-on-SBP-finite-differences).
dg = DGMulti(polydeg = 3,
             element_type = Tri(),
             approximation_type = SBP(),
             surface_flux = flux_lax_friedrichs,
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

mesh = DGMultiMesh(dg,
                   cells_per_dimension=(32, 32), # initial_refinement_level = 5
                   coordinates_min=(-2.0, -2.0),
                   coordinates_max=( 2.0,  2.0),
                   periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    boundary_conditions=boundary_condition_periodic)
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=10)
analysis_callback = AnalysisCallback(semi, interval=100, uEltype=real(dg))
callbacks = CallbackSet(analysis_callback, alive_callback);

sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-6, reltol=1.0e-6,
            callback=callbacks, save_everystep=false);
#-
using Plots
pd = PlotData2D(sol)
plot(pd)
#-
plot(pd["rho"])
plot!(getmesh(pd))


# ## Triangular meshes on non-Cartesian domains
# To use triangular meshes on a non-Cartesian domain, Trixi uses the package [StartUpDG.jl](https://github.com/jlchan/StartUpDG.jl).
# The following example is based on [`elixir_euler_triangulate_pkg_mesh.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/dgmulti_2d/elixir_euler_triangulate_pkg_mesh.jl)
# and uses a pre-defined mesh from StartUpDG.jl.
using Trixi, OrdinaryDiffEq

# We want to simulate the smooth initial condition [`initial_condition_convergence_test`](@ref)
# with source terms [`source_terms_convergence_test`](@ref) for the 2D compressible Euler equations.
equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

# We create the solver `DGMulti` with triangular elements (`Tri()`) as before.
dg = DGMulti(polydeg = 3, element_type = Tri(),
             surface_flux = flux_lax_friedrichs,
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

# StartUpDG.jl provides for instance a pre-defined Triangulate geometry for a rectangular domain with
# hole. Other pre-defined Triangulate geometries are e.g., `SquareDomain`, `RectangularDomainWithHole`,
# `Scramjet`, and `CircularDomain`.
meshIO = StartUpDG.triangulate_domain(StartUpDG.RectangularDomainWithHole());

# The pre-defined Triangulate geometry in StartUpDG has integer boundary tags. With [`DGMultiMesh`](@ref)
# we assign boundary faces based on these integer boundary tags and create a mesh compatible with Trixi.
mesh = DGMultiMesh(meshIO, dg, Dict(:bottom=>1, :right=>2, :top=>3, :left=>4))

boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; :bottom => boundary_condition_convergence_test,
                         :right => boundary_condition_convergence_test,
                         :top => boundary_condition_convergence_test,
                         :left => boundary_condition_convergence_test)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=20)
analysis_callback = AnalysisCallback(semi, interval=200, uEltype=real(dg))
callbacks = CallbackSet(alive_callback, analysis_callback);

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), # TODO BB: anderes Zeitintegrationsverfahren wie oben immer?
            dt = 0.5 * estimate_dt(mesh, dg), save_everystep=false, callback=callbacks);
#-
using Plots
pd = PlotData2D(sol)
plot(pd["rho"])
plot!(getmesh(pd))

# For more information, please have a look in the [StartUpDG.jl documentation](https://jlchan.github.io/StartUpDG.jl/stable/).


# ## Other methods via [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl)
# The `DGMulti` solver also supports other methods than DGSEM. The important property a method has to
# fulfill is the summation-by-parts (SBP) property. The package [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl)
# provides such methods, like a finite difference SBP (FD SBP) scheme. To do this,
# you need create an SBP derivative operator and pass that as `approximation_type`
# to the `DGMulti` constructor. For example, the classical second-order FD SBP operator
# can be created as
using Trixi.SummationByPartsOperators # or add SummationByPartsOperators to your project and use it directly
D = derivative_operator(MattssonNordström2004(), derivative_order=1, accuracy_order=2,
                        xmin=0.0, xmax=1.0, N=11)
# Here, the arguments `xmin` and `xmax` do not matter beyond setting the real type
# used for the operator - they just set a reference element and are rescaled on the
# physical elements. The parameter `N` determines the number of finite difference nodes.
# Then, `D` can be used as `approximation_type` like `SBP()` in a multi-block fashion.
# In multiple dimensions, such a 1D SBP operator will be used in a tensor product fashion,
# i.e., in each coordinate direction. In particular, you can use them only on 1D, 2D `Quad()`,
# and 3D `Hex()` elements.
#
# You can also use fully periodic single-block FD methods by creating a periodic SBP
# operator. For example, a fully periodic FD operator can be constructed as
D = periodic_derivative_operator(derivative_order=1, accuracy_order=2,
                                 xmin=0.0, xmax=1.0, N=11)
# An example using such an FD method is implemented in
# [`elixir_euler_fdsbp_periodic.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/dgmulti_2d/elixir_euler_fdsbp_periodic.jl).
# For all parameters and other calling options, please have a look in the
# [documentation of SummationByPartsOperators.jl](https://ranocha.de/SummationByPartsOperators.jl/stable/).

# Another possible method is for instance a continuous Galerkin (CGSEM) method. You can use such a
# method with polynomial degree of `3` (`N=4` Legendre Lobatto nodes on `[0, 1]`) coupled continuously
# on a uniform mesh with `Nx=10` elements by setting `approximation_type` to
using Trixi.SummationByPartsOperators # or add SummationByPartsOperators to your project and use it directly
D = couple_continuously(legendre_derivative_operator(xmin=0.0, xmax=1.0, N=4),
                        UniformPeriodicMesh1D(xmin=-1.0, xmax=1.0, Nx=10))

# To choose a discontinuous coupling (DGSEM), use `couple_discontinuously()` instead of `couple_continuously()`.

# For more information and other SBP operators, see the documentations of [StartUpDG.jl](https://jlchan.github.io/StartUpDG.jl/dev/)
# and [SummationByPartsOperators.jl](https://ranocha.de/SummationByPartsOperators.jl/stable/).
