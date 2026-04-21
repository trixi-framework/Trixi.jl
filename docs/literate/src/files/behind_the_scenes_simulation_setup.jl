#src # Behind the scenes of a simulation setup

# This tutorial will guide you through a simple Trixi.jl setup ("elixir"), giving an overview of
# what happens in the background during the initialization of a simulation. While the setup
# described herein does not cover all details, it involves relatively stable parts of Trixi.jl that
# are unlikely to undergo significant changes in the near future. The goal is to clarify some of
# the more fundamental, *technical* concepts that are applicable to a variety of
# (also more complex) configurations.

# Trixi.jl follows the [method of lines](http://www.scholarpedia.org/article/Method_of_lines) concept for solving partial differential equations (PDEs).
# Firstly, the PDEs are reduced to a (potentially huge) system of
# ordinary differential equations (ODEs) by discretizing the spatial derivatives. Subsequently,
# these generated ODEs may be solved with methods available in OrdinaryDiffEq.jl or those specifically
# implemented in Trixi.jl. The following steps elucidate the process of transitioning from PDEs to
# ODEs within the framework of Trixi.jl.

# ## Basic setup

# Import essential libraries and specify an equation.

using Trixi, OrdinaryDiffEq
equations = LinearScalarAdvectionEquation2D((-0.2, 0.7))

# Generate a spatial discretization using a [`TreeMesh`](@ref) with a pre-coarsened set of cells.

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)

coarsening_patches = ((type = "box", coordinates_min = [0.0, -2.0],
                       coordinates_max = [2.0, 0.0]),)

mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level = 2,
                n_cells_max = 30_000,
                coarsening_patches = coarsening_patches)

# The created `TreeMesh` looks like the following:

# ![TreeMesh_example](https://github.com/trixi-framework/Trixi.jl/assets/119304909/d5ef76ee-8246-4730-a692-b472c06063a3)

# Instantiate a [`DGSEM`](@ref) solver with a user-specified polynomial degree. The solver
# will define `polydeg + 1` [Gauss-Lobatto nodes](https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Lobatto_rules) and their associated weights within
# the reference interval ``[-1, 1]`` in each spatial direction. These nodes will be subsequently
# used to approximate solutions on each leaf cell of the `TreeMesh`.

solver = DGSEM(polydeg = 3)

# Gauss-Lobatto nodes with `polydeg = 3`:

# ![Gauss-Lobatto_nodes_example](https://github.com/trixi-framework/Trixi.jl/assets/119304909/1d894611-801e-4f75-bff0-d77ca1c672e5)

# ## Overview of the [`SemidiscretizationHyperbolic`](@ref) type

# At this stage, all necessary components for configuring the spatial discretization are in place.
# The remaining task is to combine these components into a single structure that will be used
# throughout the entire simulation process. This is where [`SemidiscretizationHyperbolic`](@ref)
# comes into play.

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

# The constructor for the `SemidiscretizationHyperbolic` object calls numerous sub-functions to
# perform the necessary initialization steps. A brief description of the key sub-functions is
# provided below.

# - `init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)`

#   The fundamental elements for approximating the solution are the leaf
#   cells. The solution is constructed as a polynomial of the degree specified in the `DGSEM`
#   solver in each spatial direction on each leaf cell. This polynomial approximation is evaluated 
#   at the Gauss-Lobatto nodes mentioned earlier. The `init_elements` function extracts
#   these leaf cells from the `TreeMesh`, assigns them the label "elements", records their
#   coordinates, and maps the Gauss-Lobatto nodes from the 1D interval ``[-1, 1]`` onto each coordinate axis
#   of every element.

#   ![elements_example](https://github.com/trixi-framework/Trixi.jl/assets/119304909/9f486670-b579-4e42-8697-439540c8bbb4)

#   The visualization of elements with nodes shown here includes spaces between elements, which do
#   not exist in reality. This spacing is included only for illustrative purposes to underscore the
#   separation of elements and the independent projection of nodes onto each element.

# - `init_interfaces(leaf_cell_ids, mesh, elements)`

#   At this point, the elements with nodes have been defined; however, they lack the necessary
#   communication functionality. This is crucial because the local solution polynomials on the
#   elements are not independent of each other. Furthermore, nodes on the boundary of adjacent
#   elements share the same spatial location, which requires a method to combine this into a
#   meaningful solution.
#   Here [Riemann solvers](https://en.wikipedia.org/wiki/Riemann_solver#Approximate_solvers)
#   come into play which can handle the principal ambiguity of a multi-valued solution at the
#   same spatial location.

#   As demonstrated earlier, the elements can have varying sizes. Let us initially consider
#   neighbors with equal size. For these elements, the `init_interfaces` function generates
#   interfaces that store information about adjacent elements, their relative positions, and
#   allocate containers for sharing solution data between neighbors during the solution process. 

#   In our visualization, these interfaces would conceptually resemble tubes connecting the
#   corresponding elements.

#   ![interfaces_example](https://github.com/trixi-framework/Trixi.jl/assets/119304909/bc3b6b02-afbc-4371-aaf7-c7bdc5a6c540)

# - `init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)`

#   Returning to the consideration of different sizes among adjacent elements, within the
#   `TreeMesh`, adjacent leaf cells can vary in side length by a maximum factor of two. This
#   implies that a large element has one neighbor of
#   equal size with a connection through an interface, or two neighbors at half the size,
#   requiring a connection through so called "mortars". In 3D, a large element would have
#   four small neighbor elements.

#   Mortars store information about the connected elements, their relative positions, and allocate
#   containers for storing the solutions along the boundaries between these elements.

#   Due to the differing sizes of adjacent elements, it is not feasible to directly map boundary
#   nodes of adjacent elements. Therefore, the concept of mortars employs a mass-conserving
#   interpolation function to map boundary nodes from a larger element to a smaller one.

#   In our visualization, mortars are represented as branched tubes.

#   ![mortars_example](https://github.com/trixi-framework/Trixi.jl/assets/119304909/43a95a60-3a31-4b1f-8724-14049e7a0481)

# - `init_boundaries(leaf_cell_ids, mesh, elements)`

#   In order to apply boundary conditions, it is necessary to identify the locations of the
#   boundaries. Therefore, we initialize a "boundaries" object, which records the elements that
#   contain boundaries, specifies which side of an element is a boundary, stores the coordinates
#   of boundary nodes, and allocates containers for managing solutions at these boundaries.

#   In our visualization, boundaries and their corresponding nodes are highlighted with green,
#   semi-transparent lines.

#   ![boundaries_example](https://github.com/trixi-framework/Trixi.jl/assets/119304909/21996b20-4a22-4dfb-b16a-e2c22c2f29fe)

# All the structures mentioned earlier are collected as a cache of type `NamedTuple`. Subsequently,
# an object of type `SemidiscretizationHyperbolic` is initialized using this cache, initial and
# boundary conditions, equations, mesh and solver.

# In conclusion, the primary purpose of a `SemidiscretizationHyperbolic` is to collect equations,
# the geometric representation of the domain, and approximation instructions, creating specialized
# structures to interconnect these components in a manner that enables their utilization for
# the numerical solution of partial differential equations (PDEs).

# As evident from the earlier description of `SemidiscretizationHyperbolic`, it comprises numerous
# functions called subsequently. Without delving into details, the structure of the primary calls
# are illustrated as follows:

# ![SemidiscretizationHyperbolic_structure](https://github.com/trixi-framework/Trixi.jl/assets/119304909/8bf59422-0537-4d7a-9f13-d9b2253c19d7)

# ## Overview of the [`semidiscretize`](@ref) function

# At this stage, we have defined the equations and configured the domain's discretization. The
# final step before solving is to select a suitable time span and apply the corresponding initial
# conditions, which are already stored in the initialized `SemidiscretizationHyperbolic` object.

# The purpose of the [`semidiscretize`](@ref) function is to wrap the semidiscretization as an
# `ODEProblem` within the specified time interval. During this procedure the approximate solution
#  is created at the given initial time via the specified `initial_condition` function from the 
#  `SemidiscretizationHyperbolic` object. This `ODEProblem` can be subsequently passed to the
# `solve` function from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package
# or to [`Trixi.solve`](@ref).

ode = semidiscretize(semi, (0.0, 1.0));

# The `semidiscretize` function involves a deep tree of subsequent calls, with the primary ones
# explained below.

# - `allocate_coefficients(mesh, equations, solver, cache)`

#   To apply initial conditions, a data structure ("container") needs to be generated to store the
#   initial values of the target variables for each node within each element.

#   Since only one-dimensional `Array`s are `resize!`able in Julia, we use `Vector`s as an internal
#   storage for the target variables and `resize!` them whenever needed, e.g. to change the number
#   of elements. Then, during the solving process the same memory is reused by `unsafe_wrap`ping
#   multi-dimensional `Array`s around the internal storage.

# - `wrap_array(u_ode, semi)`

#   As previously noted, `u_ode` is constructed as a 1D vector to ensure compatibility with
#   OrdinaryDiffEq.jl. However, for internal use within Trixi.jl, identifying which part of the
#   vector relates to specific variables, elements, or nodes can be challenging.

#   This is why the `u_ode` vector is wrapped by the `wrap_array` function using `unsafe_wrap`
#   to form a multidimensional array `u`. In this array, the first dimension corresponds to
#   variables, followed by N dimensions corresponding to nodes for each of N space dimensions.
#   The last dimension corresponds to the elements. 
#   Consequently, navigation within this multidimensional array becomes noticeably easier.

#   "Wrapping" in this context involves the creation of a reference to the same storage location
#   but with an alternative structural representation. This approach enables the use of both
#   instances `u` and `u_ode` as needed, so that changes are simultaneously reflected in both.
#   This is possible because, from a storage perspective, they share the same stored data, while
#   access to this data is provided in different ways.

# - `compute_coefficients!(u, initial_conditions, t, mesh::DG, equations, solver, cache)`

#   Now the variable `u`, intended to store solutions, has been allocated and wrapped, it is time
#   to apply the initial conditions. The `compute_coefficients!` function calculates the initial
#   conditions for each variable at every node within each element and properly stores them in the
#   `u` array.

# At this stage, the `semidiscretize` function has all the necessary components to initialize and
# return an `ODEProblem` object, which will be used by the `solve` function to compute the
# solution.

# In summary, the internal workings of `semidiscretize` with brief descriptions can be presented
# as follows.

# ![semidiscretize_structure](https://github.com/trixi-framework/Trixi.jl/assets/119304909/491eddc4-aadb-4e29-8c76-a7c821d0674e)

# ## Functions `solve` and `rhs!`

# Once the `ODEProblem` object is initialized, the `solve` function and one of the ODE solvers from
# the OrdinaryDiffEq.jl package can be utilized to compute an approximated solution using the
# instructions contained in the `ODEProblem` object.

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false), dt = 0.01,
            save_everystep = false);

# Since the `solve` function and the ODE solver have no knowledge
# of a particular spatial discretization, it is necessary to define a
# "right-hand-side function", `rhs!`, within Trixi.jl.

# Trixi.jl includes a set of `rhs!` functions designed to compute `du`, i.e.,
# ``\frac{\partial u}{\partial t}`` according to the structure
# of the setup. These `rhs!` functions calculate interface, mortars, and boundary fluxes, in
# addition to surface and volume integrals, in order to construct the `du` vector. This `du` vector
# is then used by the time integration method to obtain the solution at the subsequent time step.
# The `rhs!` function is called by time integration methods in each iteration of the solve loop
# within OrdinaryDiffEq.jl, with arguments `du`, `u`, `semidiscretization`, and the current time.

# Trixi.jl uses a two-levels approach for `rhs!` functions. The first level is limited to a
# single function for each `semidiscretization` type, and its role is to redirect data to the
# target `rhs!` for specific solver and mesh types. This target `rhs!` function is responsible
# for calculating `du`.

# Path from the `solve` function call to the appropriate `rhs!` function call:

# ![rhs_structure](https://github.com/trixi-framework/Trixi.jl/assets/119304909/dbea9a0e-25a4-4afa-855e-01f1ad619982)

# Computed solution:

using Plots
plot(sol)
pd = PlotData2D(sol)
plot!(getmesh(pd))
