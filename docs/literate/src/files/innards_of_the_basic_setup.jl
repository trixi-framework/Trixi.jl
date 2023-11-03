#src # Core aspects of the basic setup

# This tutorial aims to provide insights into the Trixi.jl framework, with a focus on commonly
# used complex functions. These functions may lack clarity in their names and brief documentation
# descriptions.

# We will guide you through a simplified setup, emphasizing functionalities that may not be evident
# to an average user. This setup is based on stable parts of Trixi.jl that are unlikely to undergo
# significant changes in the near future. Nevertheless, it will clarify fundamental concepts that
# continue to be applied in more recent and flexible configurations.


# ## Basic setup

# Import essential libraries and specify an equation.

using Trixi, OrdinaryDiffEq
equations = inearScalarAdvectionEquation2D((-0.2, 0.7))

# Generate a spatial discretization using a [`TreeMesh`](@ref) with a pre-coarsened set of cells.

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)

coarsening_patches = ( # Coarsen cell in the lower-right quarter
  (type = "box", coordinates_min = [0.0, -2.0], coordinates_max = [2.0, 0.0]), 
)

 mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level=2, 
                 n_cells_max=30_000,
                 coarsening_patches=coarsening_patches)

# The created `TreeMesh` looks like the following:

# ![TreeMesh_example](https://github.com/trixi-framework/Trixi.jl/assets/119304909/d5ef76ee-8246-4730-a692-b472c06063a3)

# Instantiate a [`DGSEM`](@ref) solver with a user-specified polynomial degree. The solver
# will define `polydeg + 1` Gauss-Lobatto nodes and their associated weights within
# the reference interval ``[-1, 1]`` in each spatial direction. These nodes will be subsequently used to approximate solutions
# on each leaf cell of the `TreeMesh`.

solver = DGSEM(polydeg=3)

# Gauss-Lobatto nodes with `N=4`:

# ![Gauss-Lobatto_nodes_example](https://github.com/trixi-framework/Trixi.jl/assets/119304909/401e5e85-026e-48b6-8a1f-dca0306f3bb0)


# ## Overview of the [`SemidiscretizationHyperbolic`](@ref) type

# At this stage, all the necessary components for configuring domain discretization are in place.
# The remaining task is to combine these components into a single structure that will be used
# throughout the entire solving process. This is where [`SemidiscretizationHyperbolic`](@ref) comes
# into play.

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)

# The `SemidiscretizationHyperbolic` function calls numerous sub-functions to perform the necessary
# steps. A brief description of the key sub-functions is provided below.

# - `init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)`
 
#   As was already mentioned, the fundamental elements for approximating a solution are the leaf
#   cells. This implies that on each leaf cell, the solution is treated as a polynomial of the
#   degree specified in the `DGSEM` solver and evaluated at the Gauss-Lobatto nodes, which were
#   previously illustrated. To provide this, the `init_elements` function extracts these leaf cells
#   from the `TreeMesh`, assigns them the label "elements", records their coordinates, and maps the
#   Gauss-Lobatto nodes from the 1D interval [-1, 1] onto each axis of every element.

#   ![elements_example](https://github.com/trixi-framework/Trixi.jl/assets/119304909/534131bd-e85b-43d5-860d-2db6e60ce921)

#   The visualization of elements with nodes includes spaces between elements, which do not exist
#   in  reality. This spacing is included only for illustrative purposes to underscore the
#   separation of elements and the independent projection of nodes onto each element.

# - `init_interfaces(leaf_cell_ids, mesh, elements)`

#   At this point, the elements with nodes have been defined; however, they lack the necessary
#   communication functionality. This is crucial because the solutions on the elements are not
#   independent of each other. Furthermore, nodes on the boundary of adjacent elements share
#   the same spatial location, requiring a method to combine their solutions.

#   As demonstrated earlier, the elements can have varying sizes. Let us initially consider
#   neighbors with equal size. For these elements, the `init_interfaces` function generates
#   interfaces that store information about adjacent elements, their relative positions, and
#   allocate containers for sharing solutions between neighbors during the solving process. 

#   In our visualization, these interfaces would conceptually resemble tubes connecting the
#   corresponding elements.

#   ![interfaces_example](https://github.com/trixi-framework/Trixi.jl/assets/119304909/bc3b6b02-afbc-4371-aaf7-c7bdc5a6c540)

# - `init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)`

#   Returning to the consideration of different sizes among adjacent elements, within the
#   `TreeMesh`, adjacent leaf cells can vary in size by a maximum factor of two. This implies
#   that a coarsened element on each side, excluding the domain boundaries, has one neighbor of
#   equal size with a connection through an interface, or two neighbors with sizes two times
#   smaller, requiring a connection through so called "mortars".

#   Mortars store information about the connected elements, their relative positions, and allocate
#   containers for sharing solutions between these elements along their boundaries.

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

# All the structures mentioned earlier are packed as a cache of type `Tuple`. Subsequently, an
# object of type `SemidiscretizationHyperbolic` is initialized using this cache, initial and
# boundary conditions, equations, mesh and solver.

# In conclusion, `Semidiscretization`'s primary function is to collect equations, the geometric
# representation of the domain, and approximation instructions, creating specialized structures to
# interconnect these components in a manner that enables their utilization for the numerical
# solution of partial differential equations (PDEs).

# As evident from the earlier description of `SemidiscretizationHyperbolic`, it comprises numerous
# functions called recursively. Without delving into details, the structure of the primary calls
# can be illustrated as follows:

# ![SemidiscretizationHyperbolic_structure](https://github.com/trixi-framework/Trixi.jl/assets/119304909/2cdfe3d1-f88a-4028-b83c-908d34d400cd)


# ## Overview of the [`semidiscretize`](@ref) function

# At this stage, we have defined the equations and configured the domain's discretization. The
# final step before solving is to select a suitable time span and apply the corresponding initial
# conditions, which are already stored in the initialized `SemidiscretizationHyperbolic` object.

# The purpose of the [`semidiscretize`](@ref) function is to wrap the semi-discretization as an
# `ODEProblem` within the specified time interval, while also applying the initial conditions at
# the left boundary of this interval. This `ODEProblem` can be subsequently passed to the `solve`
# function from the OrdinaryDiffEq.jl package.

ode = semidiscretize(semi, (0.0, 1.0));
 
# The `semidiscretize` function involves a deep tree of recursive calls, with the primary ones
# explained below.

# - `allocate_coefficients(mesh, equations, solver, cache)`

#   To apply initial conditions, a container needs to be generated to store the initial values of
#   the target variables for each node within each element. The `allocate_coefficients` function
#   initializes `u_ode` as a 1D zero-vector with a length that depends on the number of variables,
#   elements, nodes, and dimensions. The use of a 1D vector format is consistent with the
#   requirements of the ODE-solvers from OrdinaryDiffEq.jl. Therefore, Trixi.jl follows this
#   format to be able to utilize the functionalities of OrdinaryDiffEq.jl.

# - `wrap_array(u_ode, semi)`

#   As previously noted, `u_ode` is constructed as a 1D vector to ensure compatibility with
#   OrdinaryDiffEq.jl. However, for internal use within Trixi.jl, identification which part of the
#   vector relates to specific variables, elements and nodes can be challenging.

#   This is why the `u_ode` vector is wrapped by the `wrap_array` function to create a
#   multidimensional array `u`, with each dimension representing variables, nodes and elements.
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

# Once the `ODEProblem` object is initialized, the `solve` function and one of the ODE-solvers from
# the OrdinaryDiffEq.jl package can be utilized to compute an approximated solution using the
# instructions contained in the `ODEProblem` object.

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=0.01, save_everystep=false);

# Since the `solve` function and the ODE-solver are defined in another package without knowledge
# of how to handle discretizations performed in Trixi.jl, it is necessary to define the
# right-hand-side function, `rhs!`, within Trixi.jl.

# Trixi.jl includes a set of `rhs!` functions designed to compute `du` according to the structure
# of the setup. These `rhs!` functions calculate interface, mortars, and boundary fluxes, in
# addition to surface and volume integrals, in order to construct the `du` vector. This `du` vector
# is then used by the time integration method to derive solutions in the subsequent time step.
# The `rhs!` function is called by time integration methods in each iteration of the solve-loop
# within OrdinaryDiffEq.jl, with arguments `du`, `u`, `Semidiscretization`, and the time step.

# The problem is that `rhs!` functions within Trixi.jl are specialized for specific solver and mesh
# types. However, the types of arguments passed to `rhs!` by time integration methods do not
# explicitly provide this information. Consequently, Trixi.jl uses a two-levels approach for `rhs!`
# functions. The first level is limited to a single function for each `semidiscretization` type,
# and its role is to redirect data to the target `rhs!`. It performs this by extracting the
# necessary data from the integrator and passing them, along with the originally received
# arguments, to the specialized for solver and mesh types `rhs!` function, which is
# responsible for calculating `du`.

# Path from the `solve` function call to the appropriate `rhs!` function call:

# ![rhs_structure](https://github.com/trixi-framework/Trixi.jl/assets/119304909/dbea9a0e-25a4-4afa-855e-01f1ad619982)

# Computed solution:

using Plots
plot(sol)
pd = PlotData2D(sol)
plot!(getmesh(pd))


# ## Package versions

# These results were obtained using the following versions.

using InteractiveUtils
versioninfo()

using Pkg
Pkg.status(["Trixi", "OrdinaryDiffEq", "Plots", "ForwardDiff"],
           mode=PKGMODE_MANIFEST)
