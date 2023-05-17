#src # Unstructured meshes with HOHQMesh.jl

# Trixi.jl supports numerical approximations on unstructured quadrilateral meshes
# with the [`UnstructuredMesh2D`](@ref) mesh type.

# The purpose of this tutorial is to demonstrate how to use the `UnstructuredMesh2D`
# functionality of Trixi.jl. This begins by running and visualizing an available unstructured
# quadrilateral mesh example. Then, the tutorial will demonstrate how to
# conceptualize a problem with curved boundaries, generate
# a curvilinear mesh using the available software in the Trixi.jl ecosystem,
# and then run a simulation using Trixi.jl on said mesh.

# Unstructured quadrilateral meshes can be made
# with the [High-Order Hex-Quad Mesh (HOHQMesh) generator](https://github.com/trixi-framework/HOHQMesh)
# created and developed by David Kopriva.
# HOHQMesh is a mesh generator specifically designed for spectral element methods.
# It provides high-order boundary curve information (needed to accurately set boundary conditions)
# and elements can be larger (due to the high accuracy of the spatial approximation)
# compared to traditional finite element mesh generators.
# For more information about the design and features of HOHQMesh one can refer to its
# [official documentation](https://trixi-framework.github.io/HOHQMesh/).

# HOHQMesh is incorporated into the Trixi.jl framework via the registered Julia package
# [HOHQMesh.jl](https://github.com/trixi-framework/HOHQMesh.jl).
# This package provides a Julia wrapper for the HOHQMesh generator that allows users to easily create mesh
# files without the need to build HOHQMesh from source. To install the HOHQMesh package execute
# ```julia
# import Pkg; Pkg.add("HOHQMesh")
# ```
# Now we are ready to generate an unstructured quadrilateral mesh that can be used by Trixi.jl.

# ## Running and visualizing an unstructured simulation

# Trixi.jl supports solving hyperbolic problems on several mesh types.
# There is a default example for this mesh type that can be executed by

using Trixi
redirect_stdio(stdout=devnull, stderr=devnull) do # code that prints annoying stuff we don't want to see here #hide #md
trixi_include(default_example_unstructured())
end #hide #md

# This will compute a smooth, manufactured solution test case for the 2D compressible Euler equations
# on the curved quadrilateral mesh described in the
# [Trixi.jl documentation](https://trixi-framework.github.io/Trixi.jl/stable/meshes/unstructured_quad_mesh/).

# Apart from the usual error and timing output provided by the Trixi.jl run, it is useful to visualize and inspect
# the solution. One option available in the Trixi.jl framework to visualize the solution on
# unstructured quadrilateral meshes is post-processing the
# Trixi.jl output file(s) with the [`Trixi2Vtk`](https://github.com/trixi-framework/Trixi2Vtk.jl) tool
# and plotting them with [ParaView](https://www.paraview.org/download/).

# To convert the HDF5-formatted `.h5` output file(s) from Trixi.jl into VTK format execute the following

using Trixi2Vtk
redirect_stdio(stdout=devnull, stderr=devnull) do # code that prints annoying stuff we don't want to see here #hide #md
trixi2vtk("out/solution_000180.h5", output_directory="out")
end #hide #md

# Note this step takes about 15-30 seconds as the package `Trixi2Vtk` must be precompiled and executed for the first time
# in your REPL session. The `trixi2vtk` command above will convert the solution file at the final time into a `.vtu` file
# which can be read in and visualized with ParaView. Optional arguments for `trixi2vtk` are: (1) Pointing to the `output_directory`
# where the new files will be saved; it defaults to the current directory. (2) Specifying a higher number of
# visualization nodes. For instance, if we want to use 12 uniformly spaced nodes for visualization we can execute

redirect_stdio(stdout=devnull, stderr=devnull) do # code that prints annoying stuff we don't want to see here #hide #md
trixi2vtk("out/solution_000180.h5", output_directory="out", nvisnodes=12)
end #hide #md

# By default `trixi2vtk` sets `nvisnodes` to be the same as the number of nodes specified in
# the `elixir` file used to run the simulation.

# Finally, if you want to convert all the solution files to VTK execute

redirect_stdio(stdout=devnull, stderr=devnull) do # code that prints annoying stuff we don't want to see here #hide #md
trixi2vtk("out/solution_000*.h5", output_directory="out", nvisnodes=12)
end #hide #md

# then it is possible to open the `.pvd` file with ParaView and create a video of the simulation.

# ## Creating a mesh using HOHQMesh

# The creation of an unstructured quadrilateral mesh using HOHQMesh.jl is driven by a **control file**. In this file the user dictates
# the domain to be meshed, prescribes any desired boundary curvature, the polynomial order of said boundaries, etc.
# In this tutorial we cover several basic features of the possible control inputs. For a complete discussion
# on this topic see the [HOHQMesh control file documentation](https://trixi-framework.github.io/HOHQMesh/the-control-file/).

# To begin, we provide a complete control file in this tutorial. After this we give a breakdown
# of the control file components to explain the chosen parameters.

# Suppose we want to create a mesh of a domain with straight sided
# outer boundaries and a curvilinear "ice cream cone" shaped object at its center.

# ![mesh_boundary_cartoon](https://user-images.githubusercontent.com/25242486/129603954-9788500d-bba8-49be-8e6f-7555099dbf7c.png)

# The associated `ice_cream_straight_sides.control` file is created below.
open("out/ice_cream_straight_sides.control", "w") do io
  println(io, raw"""
\begin{CONTROL_INPUT}
    \begin{RUN_PARAMETERS}
        mesh file name   = ice_cream_straight_sides.mesh
        plot file name   = ice_cream_straight_sides.tec
        stats file name  = none
        mesh file format = ISM-v2
        polynomial order = 4
        plot file format = skeleton
    \end{RUN_PARAMETERS}

    \begin{BACKGROUND_GRID}
        x0 = [-8.0, -8.0, 0.0]
        dx = [1.0, 1.0, 0.0]
        N  = [16,16,1]
    \end{BACKGROUND_GRID}

    \begin{SPRING_SMOOTHER}
        smoothing            = ON
        smoothing type       = LinearAndCrossBarSpring
        number of iterations = 25
    \end{SPRING_SMOOTHER}

\end{CONTROL_INPUT}

\begin{MODEL}

    \begin{INNER_BOUNDARIES}

        \begin{CHAIN}
            name = IceCreamCone
            \begin{END_POINTS_LINE}
                name = LeftSlant
                xStart = [-2.0, 1.0, 0.0]
                xEnd   = [ 0.0, -3.0, 0.0]
            \end{END_POINTS_LINE}

            \begin{END_POINTS_LINE}
                name = RightSlant
                xStart = [ 0.0, -3.0, 0.0]
                xEnd   = [ 2.0, 1.0, 0.0]
            \end{END_POINTS_LINE}

            \begin{CIRCULAR_ARC}
                name        = IceCream
                units       = degrees
                center      = [ 0.0, 1.0, 0.0]
                radius      = 2.0
                start angle = 0.0
                end angle   = 180.0
            \end{CIRCULAR_ARC}
        \end{CHAIN}

    \end{INNER_BOUNDARIES}

\end{MODEL}
\end{FILE}
""")
end

# The first three blocks of information are wrapped within a `CONTROL_INPUT` environment block as they define the
# core components of the quadrilateral mesh that will be generated.

# The first block of information in `RUN_PARAMETERS` is
# ```
# \begin{RUN_PARAMETERS}
#    mesh file name   = ice_cream_straight_sides.mesh
#    plot file name   = ice_cream_straight_sides.tec
#    stats file name  = none
#    mesh file format = ISM-v2
#    polynomial order = 4
#    plot file format = skeleton
# \end{RUN_PARAMETERS}
# ```

# The mesh and plot file names will be the files created by HOHQMesh once successfully executed. The stats file name is
# available if you wish to also save a collection of mesh statistics. For this example it is deactivated.
# These file names given within `RUN_PARAMETERS` **should match** that of the control file, and although this is not required by
# HOHQMesh, it is a useful style convention.
# The mesh file format `ISM-v2` in the format currently required by Trixi.jl. The `polynomial order` prescribes the order
# of an interpolant constructed on the Chebyshev-Gauss-Lobatto nodes that is used to represent any curved boundaries on a particular element.
# The plot file format of `skeleton` means that visualizing the plot file will only draw the element boundaries (and no internal nodes).
# Alternatively, the format can be set to `sem` to visualize the interior nodes of the approximation as well.

# The second block of information in `BACKGROUND_GRID` is
# ```
# \begin{BACKGROUND_GRID}
#   x0 = [-8.0, -8.0, 0.0]
#   dx = [1.0, 1.0, 0.0]
#   N  = [16,16,1]
# \end{BACKGROUND_GRID}
# ```

# This lays a grid of Cartesian elements for the domain beginning at the point `x0` as its bottom-left corner.
# The value of `dx`, which could differ in each direction if desired, controls the step size taken in each Cartesian direction.
# The values in `N` set how many Cartesian box elements are set in each coordinate direction.
# The above parameters define a $16\times 16$ element square mesh on $[-8,8]^2$.
# Further, this sets up four outer boundaries of the domain that are given the default names: `Top, Left, Bottom, Right`.

# The third block of information in `SPRING_SMOOTHER` is
# ```
# \begin{SPRING_SMOOTHER}
#    smoothing            = ON
#    smoothing type       = LinearAndCrossBarSpring
#    number of iterations = 25
# \end{SPRING_SMOOTHER}
# ```

# Once HOHQMesh generates the mesh, a spring-mass-dashpot model is created to smooth the mesh and create "nicer" quadrilateral elements.
# The [default parameters of Hooke's law](https://trixi-framework.github.io/HOHQMesh/the-control-input/#the-smoother)
# for the spring-mass-dashpot model have been selected after a fair amount of experimentation across many meshes.
# If you wish to deactivate this feature you can set `smoothing = OFF` (or remove this block from the control file).

# After the `CONTROL_INPUT` environment block comes the `MODEL` environment block. It is here where the user prescribes curved boundary information with either:
# * An `OUTER_BOUNDARY` (covered in the next section of this tutorial).
# * One or more `INNER_BOUNDARIES`.

# There are several options to describe the boundary curve data to HOHQMesh like splines or parametric curves.

# For the example `ice_cream_straight_sides.control` we define three internal boundaries; two straight-sided and
# one as a circular arc.
# Within the HOHQMesh control input each curve must be assigned to a `CHAIN` as shown below in the complete
# `INNER_BOUNDARIES` block.
# ```
# \begin{INNER_BOUNDARIES}
#
#    \begin{CHAIN}
#    name = IceCreamCone
#    \begin{END_POINTS_LINE}
#       name = LeftSlant
#       xStart = [-2.0, 1.0, 0.0]
#       xEnd   = [ 0.0, -3.0, 0.0]
#    \end{END_POINTS_LINE}
#
#    \begin{END_POINTS_LINE}
#       name = RightSlant
#       xStart = [ 0.0, -3.0, 0.0]
#       xEnd   = [ 2.0, 1.0, 0.0]
#    \end{END_POINTS_LINE}
#
#    \begin{CIRCULAR_ARC}
#       name        = IceCream
#       units       = degrees
#       center      = [ 0.0, 1.0, 0.0]
#       radius      = 2.0
#       start angle = 0.0
#       end angle   = 180.0
#    \end{CIRCULAR_ARC}
#    \end{CHAIN}
#
# \end{INNER_BOUNDARIES}
# ```

# It is important to note there are two `name` quantities one for the `CHAIN` and one for the `PARAMETRIC_EQUATION_CURVE`.
# The name for the `CHAIN` is used internally by HOHQMesh, so if you have multiple `CHAIN`s they **must be given a unique name**.
# The name for the `PARAMETRIC_EQUATION_CURVE` will be printed to the appropriate boundaries within the `.mesh` file produced by
# HOHQMesh.

# We create the mesh file `ice_cream_straight_sides.mesh` and its associated file for plotting
# `ice_cream_straight_sides.tec` by using HOHQMesh.jl's function `generate_mesh`.
using HOHQMesh
control_file = joinpath("out", "ice_cream_straight_sides.control")
output = generate_mesh(control_file);

# The mesh file `ice_cream_straight_sides.mesh` and its associated file for plotting
# `ice_cream_straight_sides.tec` are placed in the `out` folder.
# The resulting mesh generated by HOHQMesh.jl is given in the following figure.

# ![mesh_straight_sides](https://user-images.githubusercontent.com/25242486/129603958-08e4b874-53d5-4511-9a54-6daf4c21edca.png)

# We note that Trixi.jl uses the boundary name information from the control file
# to assign boundary conditions in an elixir file.
# Therefore, the name should start with a letter and consist only of alphanumeric characters and underscores. Please note that the name will be treated as case sensitive.

# ## Example simulation on `ice_cream_straight_sides.mesh`

# With this newly generated mesh we are ready to run a Trixi.jl simulation on an unstructured quadrilateral mesh.
# For this we must create a new elixir file.

# The elixir file given below creates an initial condition for a
# uniform background flow state with a free stream Mach number of 0.3.
# A focus for this part of the tutorial is to specify the boundary conditions and to construct the new mesh from the
# file that was generated in the previous exercise.

# It is straightforward to set the different boundary
# condition types in an elixir by assigning a particular function to a boundary name inside a
# Julia dictionary, `Dict`, variable. Observe that the names of these boundaries match those provided by HOHQMesh
# either by default, e.g. `Bottom`, or user assigned, e.g. `IceCream`. For this problem setup use
# * Freestream boundary conditions on the four box edges.
# * Free slip wall boundary condition on the interior curved boundaries.

# To construct the unstructured quadrilateral mesh from the HOHQMesh file we point to the appropriate location
# with the variable `mesh_file` and then feed this into the constructor for the [`UnstructuredMesh2D`](@ref) type in Trixi.jl

# ```julia
# # create the unstructured mesh from your mesh file
# using Trixi
# mesh_file = joinpath("out", "ice_cream_straight_sides.mesh")
# mesh = UnstructuredMesh2D(mesh_file);
# ```

# The complete elixir file for this simulation example is given below.

using OrdinaryDiffEq, Trixi

equations = CompressibleEulerEquations2D(1.4) # set gas gamma = 1.4

## freestream flow state with Ma_inf = 0.3
@inline function uniform_flow_state(x, t, equations::CompressibleEulerEquations2D)

  ## set the freestream flow parameters
  rho_freestream = 1.0
  u_freestream = 0.3
  p_freestream = inv(equations.gamma)

  theta = 0.0 # zero angle of attack
  si, co = sincos(theta)
  v1 = u_freestream * co
  v2 = u_freestream * si

  prim = SVector(rho_freestream, v1, v2, p_freestream)
  return prim2cons(prim, equations)
end

## initial condition
initial_condition = uniform_flow_state

## boundary condition types
boundary_condition_uniform_flow = BoundaryConditionDirichlet(uniform_flow_state)

## boundary condition dictionary
boundary_conditions = Dict( :Bottom     => boundary_condition_uniform_flow,
                            :Top        => boundary_condition_uniform_flow,
                            :Right      => boundary_condition_uniform_flow,
                            :Left       => boundary_condition_uniform_flow,
                            :LeftSlant  => boundary_condition_slip_wall,
                            :RightSlant => boundary_condition_slip_wall,
                            :IceCream   => boundary_condition_slip_wall );

## DGSEM solver.
##    1) polydeg must be >= the polynomial order set in the HOHQMesh control file to guarantee
##       freestream preservation. As a extra task try setting polydeg=3
##    2) VolumeIntegralFluxDifferencing with central volume flux is activated
##       for dealiasing
volume_flux = flux_ranocha
solver = DGSEM(polydeg=4, surface_flux=flux_hll,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

## create the unstructured mesh from your mesh file
mesh_file = joinpath("out", "ice_cream_straight_sides.mesh")
mesh = UnstructuredMesh2D(mesh_file)

## Create semidiscretization with all spatial discretization-related components
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

## Create ODE problem from semidiscretization with time span from 0.0 to 2.0
tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)


## Create the callbacks to output solution files and adapt the time step
summary_callback = SummaryCallback()
save_solution = SaveSolutionCallback(interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true)
stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback, save_solution, stepsize_callback)

redirect_stdio(stdout=devnull, stderr=devnull) do # code that prints annoying stuff we don't want to see here #hide #md
## Evolve ODE problem in time using `solve` from OrdinaryDiffEq
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
## print the timer summary
summary_callback()
end #hide #md

# Visualization of the solution is carried out in a similar way as above. That is, one converts the `.h5`
# output files with `trixi2vtk` and then plot the solution in ParaView. An example plot of the pressure
# at the final time is shown below.

# ![simulation_straight_sides](https://user-images.githubusercontent.com/25242486/129733926-6ef80676-779b-4f1e-9826-3ebf750cf382.png)

# ## Making a mesh with a curved outer boundary

# Let us modify the mesh from the previous task and place a circular outer boundary instead
# of straight-sided outer boundaries.
# Note, the "ice cream cone" shape is still placed at the center of the domain.

# We create the new control file `ice_cream_curved_sides.control` file below and will then highlight the
# major differences compared to `ice_cream_straight_sides.control`.
open("out/ice_cream_curved_sides.control", "w") do io
  println(io, raw"""
\begin{CONTROL_INPUT}
    \begin{RUN_PARAMETERS}
        mesh file name   = ice_cream_curved_sides.mesh
        plot file name   = ice_cream_curved_sides.tec
        stats file name  = none
        mesh file format = ISM-v2
        polynomial order = 4
        plot file format = skeleton
    \end{RUN_PARAMETERS}

    \begin{BACKGROUND_GRID}
        background grid size = [1.0, 1.0, 0.0]
    \end{BACKGROUND_GRID}

    \begin{SPRING_SMOOTHER}
        smoothing            = ON
        smoothing type       = LinearAndCrossBarSpring
        number of iterations = 25
    \end{SPRING_SMOOTHER}

\end{CONTROL_INPUT}

\begin{MODEL}

    \begin{OUTER_BOUNDARY}
        \begin{PARAMETRIC_EQUATION_CURVE}
            name = OuterCircle
            xEqn = x(t) = 8.0*sin(2.0*pi*t)
            yEqn = y(t) = 8.0*cos(2.0*pi*t)
            zEqn = z(t) = 0.0
        \end{PARAMETRIC_EQUATION_CURVE}

    \end{OUTER_BOUNDARY}

    \begin{INNER_BOUNDARIES}

        \begin{CHAIN}
            name = IceCreamCone
            \begin{END_POINTS_LINE}
                name = LeftSlant
                xStart = [-2.0, 1.0, 0.0]
                xEnd   = [ 0.0, -3.0, 0.0]
            \end{END_POINTS_LINE}

            \begin{END_POINTS_LINE}
                name = RightSlant
                xStart = [ 0.0, -3.0, 0.0]
                xEnd   = [ 2.0, 1.0, 0.0]
            \end{END_POINTS_LINE}

            \begin{CIRCULAR_ARC}
                name        = IceCream
                units       = degrees
                center      = [ 0.0, 1.0, 0.0]
                radius      = 2.0
                start angle = 0.0
                end angle   = 180.0
            \end{CIRCULAR_ARC}
        \end{CHAIN}

    \end{INNER_BOUNDARIES}

\end{MODEL}
\end{FILE}
""")
end

# The first alteration is that we have altered the second block of information
# `BACKGROUND_GRID` within the `CONTROL_INPUT` to be
# ```
# \begin{BACKGROUND_GRID}
#    background grid size = [1.0, 1.0, 0.0]
# \end{BACKGROUND_GRID}
# ```
# This mesh control file has an outer boundary that determines the extent of the domain to be meshed.
# Therefore, we only need to supply the `background grid size` to the `BACKGROUND_GRID` control input.

# The second alteration is that the `MODEL` now contains information for an `OUTER_BOUNDARY`.
# In this case it is a circle of radius `8` centered at `[0.0, 0.0, 0.0]` written as a set of
# `PARAMETRIC_EQUATION_CURVE`s.
# ```
#    \begin{OUTER_BOUNDARY}
#
#       \begin{PARAMETRIC_EQUATION_CURVE}
#          name = OuterCircle
#          xEqn = x(t) = 8.0*sin(2.0*pi*t)
#          yEqn = y(t) = 8.0*cos(2.0*pi*t)
#          zEqn = z(t) = 0.0
#       \end{PARAMETRIC_EQUATION_CURVE}
#
#    \end{OUTER_BOUNDARY}
# ```
# Just as with the inner boundary curves, we must assign a name to the `OUTER_BOUNDARY`. It will be included
# in the generated `.mesh` file and is used within the Trixi.jl elixir file to set boundary conditions.

# Again, we create the `.mesh` and `.tec` files with HOHQMesh.jl's function `generate_mesh`
control_file = joinpath("out", "ice_cream_curved_sides.control")
output = generate_mesh(control_file);
# The files are placed in the `out` folder.

# The resulting mesh generated by HOHQMesh.jl is given in the following figure.

# ![mesh_curved_sides](https://user-images.githubusercontent.com/25242486/129603957-6a92618f-9ed8-4072-b6ab-05533bea746a.png)

# ## Running Trixi.jl on `ice_cream_curved_sides.mesh`

# We can reuse much of the elixir file to setup the uniform flow over an ice cream cone from the
# previous part of this tutorial. The only component of the elixir file that must be changed is the boundary condition
# dictionary because we now have a boundary named `OuterCircle` instead of four edges of a bounding box.

## boundary condition dictionary
boundary_conditions = Dict( :OuterCircle => boundary_condition_uniform_flow,
                            :LeftSlant   => boundary_condition_slip_wall,
                            :RightSlant  => boundary_condition_slip_wall,
                            :IceCream    => boundary_condition_slip_wall );

# Also, we must update the construction of the mesh from our new mesh file `ice_cream_curved_sides.mesh` that
# is located in the `out` folder.

## create the unstructured mesh from your mesh file
mesh_file = joinpath("out", "ice_cream_curved_sides.mesh")
mesh = UnstructuredMesh2D(mesh_file);


# We can then post-process the solution file at the final time on the new mesh with `Trixi2Vtk` and visualize with ParaView.

# ![simulation_curved_sides](https://user-images.githubusercontent.com/25242486/129733924-778795c1-9119-419a-8b89-bcbe13e33cd7.png)


# ## Setting up a simulation with AMR via `P4estMesh`
# The above explained mesh file format of `ISM-V2` only works with `UnstructuredMesh2D` and so does
# not support AMR. On the other hand, the mesh type [`P4estMesh`](@ref) allows AMR. The mesh
# constructor for the `P4estMesh` imports an unstructured, conforming mesh from an Abaqus mesh file
# (`.inp`).

# As described above, the first block of the HOHQMesh control file contains the parameter
# `mesh file format`. If you set `mesh file format = ABAQUS` instead of `ISM-V2`,
# HOHQMesh.jl's function `generate_mesh` creates an Abaqus mesh file `.inp`.
# ```julia
# using HOHQMesh
# control_file = joinpath("out", "ice_cream_straight_sides.control")
# output = generate_mesh(control_file);
# ```

# Now, you can create a `P4estMesh` from your mesh file. It is described in detail in the
# [P4est-based mesh](https://trixi-framework.github.io/Trixi.jl/stable/meshes/p4est_mesh/#P4est-based-mesh)
# part of the Trixi.jl docs.
# ```julia
# using Trixi
# mesh_file = joinpath("out", "ice_cream_straight_sides.inp")
# mesh = P4estMesh{2}(mesh_file)
# ```

# Since `P4estMesh` supports AMR, we just have to extend the setup from the first example by the
# standard AMR procedure. For more information about AMR in Trixi.jl, see the [matching tutorial](@ref adaptive_mesh_refinement).

# ```julia
# amr_indicator = IndicatorLöhner(semi, variable=density)

# amr_controller = ControllerThreeLevel(semi, amr_indicator,
#                                       base_level=0,
#                                       med_level =1, med_threshold=0.05,
#                                       max_level =3, max_threshold=0.1)

# amr_callback = AMRCallback(semi, amr_controller,
#                            interval=5,
#                            adapt_initial_condition=true,
#                            adapt_initial_condition_only_refine=true)

# callbacks = CallbackSet(..., amr_callback)
# ```

# We can then post-process the solution file at the final time on the new mesh with `Trixi2Vtk` and visualize
# with ParaView, see the appropriate [visualization section](https://trixi-framework.github.io/Trixi.jl/stable/visualization/#Trixi2Vtk)
# for details.

# ![simulation_straight_sides_p4est_amr](https://user-images.githubusercontent.com/74359358/168049930-8abce6ac-cd47-4d04-b40b-0fa459bbd98d.png)
