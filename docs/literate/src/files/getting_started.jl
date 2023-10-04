#src # Getting started

# Trixi.jl is a numerical simulation framework for conservation laws and
# is written in the [Julia programming language](https://julialang.org/).
# This tutorial is intended for beginners in Julia and Trixi.jl.
# After reading it, you will know how to install Julia and Trixi.jl on your computer,
# and you will be able to download setup files from our GitHub repository, modify them,
# and run simulations.

# ## Julia installation

# Trixi.jl works with the current stable Julia release. More information about Julia support can be
# found in the [`README.md`](https://github.com/trixi-framework/Trixi.jl#installation) file.
# A detailed description of the installation process can be found in the
# [Julia installation instructions](https://julialang.org/downloads/platform/).
# But you can follow also our short installation guidelines for Windows and Linux below.

# ### Windows

# - Download Julia installer for Windows from [https://julialang.org/downloads/](https://julialang.org/downloads/). Make sure 
#   that you choose the right version of installer (64-bit or 32-bit) according to your computer.
# - Open the downloaded installer.
# - Paste an installation directory path or find it using a file manager (select `Browse`).
# - Select `Next`.
# - Check the `Add Julia to PATH` option to add Julia to the environment variables. 
#   This makes it possible to run Julia in the terminal from any directory by only typing `julia`.
# - Select `Next`, then Julia will be installed.

# Now you can verify, that Julia is installed:
# - Press `Win+R` on a keyboard.
# - Enter `cmd` in opened window.
# - Enter in a terminal `julia`. 

# Then Julia will be invoked. To close Julia enter `exit()` or press `Ctrl+d`. 

# ### Linux

# - Open a terminal and navigate (using `cd`) to the directory, where you want to store Julia.
# - To install Julia, get a link to the latest version of Julia from the
#   [Julia website](https://julialang.org/downloads/), then download an archive file by executing:
#   ```shell
#   wget https://julialang-s3.julialang.org/bin/linux/... # your link goes here
#   ```
# - Unpack the downloaded file with: 
#   ```shell
#   tar xf julia-....tar.gz # your archive filename goes here
#   ```

# Now you can verify that Julia is installed entering `<Julia directory>/bin/julia`
# (e.g. `julia-1.8.5/bin/julia`) command in a terminal. `<Julia directory>` is the directory where
# Julia is installed.
# Then Julia will be invoked. To close Julia, enter `exit()` or press `Ctrl+d`.

# Note, that further in the tutorial Julia will be invoked only typing `julia` in a terminal.
# To enable that, you have to add
# [Julia to the PATH](https://julialang.org/downloads/platform/#linux_and_freebsd).

# ## Trixi.jl installation

# Trixi.jl and its related tools are registered Julia packages, thus their installation
# happens inside Julia.
# For a smooth workflow experience with Trixi.jl, you need to install 
# [Trixi.jl](https://github.com/trixi-framework/Trixi.jl),
# [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) and 
# [Plots.jl](https://github.com/JuliaPlots/Plots.jl).

# - Open a terminal and invoke Julia (Windows and Linux: type `julia`).
# - Execute following commands:
#   ```julia
#   import Pkg
#   Pkg.add(["OrdinaryDiffEq", "Plots", "Trixi"])
#   ```

# Now you have installed all these 
# packages. [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) provides time integration schemes
# used by Trixi.jl and [Plots.jl](https://github.com/JuliaPlots/Plots.jl) can be used to directly
# visualize Trixi.jl results from the Julia REPL.

# ## Usage

# ### Running a simulation

# To get you started, Trixi.jl has a large set
# of [example setups](https://github.com/trixi-framework/Trixi.jl/tree/main/examples), that can be taken
# as a basis for your future investigations.
# In Trixi.jl, we call these setup files "elixirs", since they contain Julia code that
# takes parts of Trixi.jl and combines them into something new.

# Now execute one of the examples using the [`trixi_include`](@ref)
# function. `trixi_include(...)` expects
# a single string argument with a path to a file containing Julia code.
# `joinpath(...)` join a path components into a full path. 
# The [`examples_dir`](@ref) function returns a path to the
# [`examples`](https://github.com/trixi-framework/Trixi.jl/tree/main/examples) folder
# that has been locally downloaded while installing Trixi.jl.

# Let's execute a short two-dimensional problem setup. It approximates the solution of
# the compressible Euler equations in 2D for an ideal gas ([`CompressibleEulerEquations2D`](@ref))
# with a weak blast wave as the initial condition.

# Start Julia in a terminal and execute following code:

# ```julia
# using Trixi, OrdinaryDiffEq
# trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"))
# ```
using Trixi, OrdinaryDiffEq #hide #md
trixi_include(@__MODULE__,joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl")) #hide #md

# To analyze the result of the computation, we can use the Plots.jl package and the function 
# `plot(...)`, which creates a graphical representation of the solution. `sol` is a variable defined in
# executed example and it contains the solution at the final moment of the simulation.

using Plots
plot(sol)

# To obtain list of all Trixi.jl elixirs execute
# [`get_examples`](@ref). It returns the path to all example setups.

get_examples()

# Editing an existing elixirs is the best way to start your first own investigation using Trixi.jl.

# ### Getting an existing setup file

# To edit an existing elixir, you first have to find a suitable one and then copy it to a local folder.
# Let's have a look how to download the
# `elixir_euler_ec.jl` elixir used in the previous section from the
# [Trixi.jl GitHub repository](https://github.com/trixi-framework/Trixi.jl).

# - All examples are located inside
#   the [`examples`](https://github.com/trixi-framework/Trixi.jl/tree/main/examples) folder.
# - Navigate to the
#   file [`elixir_euler_ec.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_euler_ec.jl).
# - Right-click the `Raw` button on the right side of the webpage and choose `Save as...` (or `Save Link As...`).
# - Choose a folder and save the file.

# ### Modifying an existing setup

# For example, we will change the initial condition for calculations that occur in
# `elixir_euler_ec.jl`. In this example we consider the compressible Euler equations in two spatial dimensions,
# ```math
# \frac{\partial}{\partial t}
# \begin{pmatrix}
# \rho \\ \rho v_1 \\ \rho v_2 \\ \rho e
# \end{pmatrix}
# +
# \frac{\partial}{\partial x}
# \begin{pmatrix}
# \rho v_1 \\ \rho v_1^2 + p \\ \rho v_1 v_2 \\ (\rho e +p) v_1
# \end{pmatrix}
# +
# \frac{\partial}{\partial y}
# \begin{pmatrix}
# \rho v_2 \\ \rho v_1 v_2 \\ \rho v_2^2 + p \\ (\rho e +p) v_2
# \end{pmatrix}
# =
# \begin{pmatrix}
# 0 \\ 0 \\ 0 \\ 0
# \end{pmatrix},
# ```
# for an ideal gas with the specific heat ratio ``\gamma``.
# Here, ``\rho`` is the density, ``v_1`` and ``v_2`` are the velocities, ``e`` is the specific total
# energy, and
# ```math
# p = (\gamma - 1) \left( \rho e - \frac{1}{2} \rho (v_1^2+v_2^2) \right)
# ```
# is the pressure.
# Initial conditions consist of initial values for ``\rho``, ``\rho v_1``,
# ``\rho v_2`` and ``\rho e``.
# One of the common initial conditions for the compressible Euler equations is a simple density wave.
# Let's implement it.

# - Open the downloaded file `elixir_euler_ec.jl` with a text editor.
# - Go to the line with the following code:
#   ```julia
#   initial_condition = initial_condition_weak_blast_wave
#   ```
#   Here, [`initial_condition_weak_blast_wave`](@ref) is used as the initial condition.
# - Comment out the line using the `#` symbol:
#   ```julia
#   # initial_condition = initial_condition_weak_blast_wave
#   ```
# - Now you can create your own initial conditions. Add the following code after the
#   commented line:

    function initial_condition_density_waves(x, t, equations::CompressibleEulerEquations2D)
        v1 = 0.1 # velocity along x-axis
        v2 = 0.2 # velocity along y-axis
        rho = 1.0 + 0.98 * sin(pi * (sum(x) - t * (v1 + v2))) # density wave profile
        p = 20 # pressure
        rho_e = p / (equations.gamma - 1) + 1/2 * rho * (v1^2 + v2^2)
        return SVector(rho, rho*v1, rho*v2, rho_e)
    end
    initial_condition = initial_condition_density_waves

# - Execute the following code one more time, but instead of `path/to/file` paste the path to the
#   `elixir_euler_ec.jl` file that you just edited.
#   ```julia
#   using Trixi
#   trixi_include(path/to/file)
#   using Plots
#   plot(sol)
#   ```
# Then you will obtain a new solution from running the simulation with a different initial condition.

trixi_include(@__MODULE__,joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"), #hide #md
 initial_condition=initial_condition) #hide #md
pd = PlotData2D(sol) #hide #md
p1 = plot(pd["rho"]) #hide #md
p2 = plot(pd["v1"], clim=(0.05, 0.15)) #hide #md
p3 = plot(pd["v2"], clim=(0.15, 0.25)) #hide #md
p4 = plot(pd["p"], clim=(10, 30)) #hide #md
plot(p1, p2, p3, p4) #hide #md

# Feel free to make further changes to the initial condition to observe different solutions.

# Now you are able to download, modify and execute simulation setups for Trixi.jl.

# ### Create first setup

# In this part of the tutorial, we will consider a creation of the first Trixi.jl setup, which is
# an extension of
# [`elixir_advection_basic.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_advection_basic.jl).
# Trixi.jl has a common basic structure of the setup, so you can create your own by extending
# the following example.

# Let's consider the linear advection equation in two-dimensional spatial domain
# [-1.0, 1.0]⨯[-1.0, 1.0] with a source term.
# ```math
# \partial_t u(t,x,y) + 0.2 \partial_x u(t,x,y) - 0.7 \partial_y u(t,x,y) = - 2 \exp(-t)
# \sin\bigl(2 \pi (x - t) \bigr) \sin\bigl(2 \pi (y - t) \bigr)
# ```
# With an initial condition
# ```math
# u(0,x,y) = \sin\bigl(\pi x \bigr) \sin\bigl(\pi y \bigr).
# ```

# The first step is to create and open a file with the .jl extension. You can do this with your
# favorite text editor.

# First you need to connect the packages that you will use in your setup. By default, you will
# always need Trixi.jl itself and [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl).

using Trixi
using OrdinaryDiffEq

# The next thing to do is to choose an equation that is suitable for your problem. To see all the
# currently implemented equations, take a look at
# [`src/equations`](https://github.com/trixi-framework/Trixi.jl/tree/main/src/equations).
# If you are interested in adding a new physics model that has not yet been implemented in
# Trixi.jl, take a look at
# [adding a new scalar conservation law](@ref adding_new_scalar_equations) and
# [adding a non-conservative equation](@ref adding_nonconservative_equation).

# The linear scalar advection equation is already implemented in Trixi.jl as
# [`LinearScalarAdvectionEquation2D`](@ref). For which we need to define a two-dimensional parameter
# named advection_velocity, suitable for our problem is (0.2, -0.7).

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# To solve our problem numerically using Trixi.jl, we have to define an instruction for spatial
# discretization. To do it, we set up a mesh. One of the widely used meshes in Trixi.jl is
# [`TreeMesh`](@ref). The spatial domain used is [-1.0, 1.0]⨯[-1.0, 1.0]. We also set a number of
# elements in the mesh using `initial_refinement_level`, which describes the initial height of the
# tree mesh. The variable `n_cells_max` is used to limit the number of elements in the mesh, which
# cannot be exceeded due to [adaptive mesh refinement](@ref Adaptive-mesh-refinement).

# All minimum and all maximum coordinates must be combined into `Tuples`.

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000)

# To approximate the solution of the defined model, we create a DG solver. The solution in each of
# the recently defined mesh elements will be approximated by a polynomial of degree `polydeg`.
# See more in the [Introduction to DG methods](@ref scalar_linear_advection_1d).

solver = DGSEM(polydeg=3)

# Now we need to define the initial conditions for our problem. All the already implemented
# initial conditions for [`LinearScalarAdvectionEquation2D`](@ref) can be found in
# [`src/equations/linear_scalar_advection_2d.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/src/equations/linear_scalar_advection_2d.jl).
# If you want to use, for example, a Gaussian pulse, it can be used as follows.
# ```julia
# initial_conditions = initial_condition_gauss
# ```
# But for our problem, we define our own initial conditions.
# ```math
# u(0,x,y) = \sin\bigl(\pi x \bigr) \sin\bigl(\pi y \bigr)
# ```
# The initial conditions function must take coordinates, time and the equation itself as arguments
# and return the initial conditions as a static vector `SVector`. Following the same structure, you
# can define your own initial conditions.

function initial_condition_sin(x, t, equations::LinearScalarAdvectionEquation2D)
    scalar = sinpi(x[1]) * sinpi(x[2])
    return SVector(scalar)
end
initial_condition = initial_condition_sin

# The next step is to define the function of the source term corresponding to our problem.
# ```math
# f(t,u,x,y) = - 2 \exp(-t) \sin\bigl(2 \pi (x - t) \bigr) \sin\bigl(2 \pi (y - t) \bigr)
# ```
# This function must take the target variable, coordinates, time and the
# equation itself as arguments and return the source term as a static vector `SVector`.

function source_term_exp_sin(u, x, t, equations::LinearScalarAdvectionEquation2D)
    scalar = - 2 * exp(-t) * sinpi(2*(x[1] - t)) * sinpi(2*(x[2] - t))
    return SVector(scalar)
end

# Now we are collecting all the information that will be needed to define spatial discretization
# and to create an ODE problem with a time span from 0.0 s to 1.0 s.

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    source_terms = source_term_exp_sin)
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

# At this point, our problem is defined. We will use the `solve` function defined in
# OrdinaryDiffEq.jl to get the solution. OrdinaryDiffEq.jl gives us ability to customize the solver
# using callbacks without actually modifying it. Trixi.jl already has some implemented
# [Callbacks](@ref callbacks-id). The most widely used callbacks in Trixi.jl are
# [step control callbacks](https://docs.sciml.ai/DiffEqCallbacks/stable/step_control/) that are
# activated at the end of each time step to perform some actions, e.g. to print a statistics.
# We will show you how to use some of the common callbacks.

# To print a summary of the simulation setup at the beginning of solve-loop
# and to reset timers we use [`SummaryCallback`](@ref).

summary_callback = SummaryCallback()

# Also we want to analyse the current state of the solution in regular intervals.
# [`AnalysisCallback`](@ref) outputs some useful statistical information during the solving process
# every `interval` time steps.

analysis_callback = AnalysisCallback(semi, interval = 5)

# It is also possible to control the time step size using [`StepsizeCallback`](@ref) if the time
# integration method isn't adaptive itself. To get more details, look at
# [CFL based step size control](@ref CFL-based-step-size-control).

stepsize_callback = StepsizeCallback(cfl = 1.6)

# To save the current numerical solution in regular intervals we use
# [`SaveSolutionCallback`](@ref). We set the interval equal 5, which means that the solution will
# be saved every 5 time steps. Also we would like to save the initial and final solutions. Solution
# will be saved as a HDF5 file located in the `out` folder. Afterwards it is possible to visualize
# the solution from the saved files using Trixi2Vtk.jl and ParaView, this is described below in the
# section [Visualize the solution](@ref Visualize-the-solution).

save_solution = SaveSolutionCallback(interval = 5,
                                     save_initial_solution = true,
                                     save_final_solution = true)

# Another useful callback is [`SaveRestartCallback`](@ref). It saves information for restarting
# in regular intervals, which we set to 100 time steps. Also we are interested in saving the
# restart file for the final solution. To perform a restart, you need to configure the restart
# setup in a special way, which is described in the section [Restart simulation](@ref restart).

save_restart = SaveRestartCallback(interval = 100, save_final_restart = true)

# Create a `CallbackSet` to collect all callbacks so that they can be passed to the `solve`
# function.

callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback, save_solution,
                        save_restart)

# The last step is to choose the time integration method, OrdinaryDiffEq.jl defines a wide range of
# [ODE solvers](https://docs.sciml.ai/DiffEqDocs/latest/solvers/ode_solve/), e.g.
# `CarpenterKennedy2N54(williamson_condition = false)`. We will pass the ODE
# problem, the ODE solver and the callbacks to the `solve` function. Also, to use
# `StepsizeCallback`, we must explicitly specify the time step `dt`, the selected value is not
# important, because it will be overwritten by `StepsizeCallback`. And there is no need to save
# every step of the solution, we are only interested in the final result.

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false), dt = 1.0,
            save_everystep = false, callback = callbacks);

# Finally, we print the timer summary.

summary_callback()

# Now you can plot the solution as shown below, analyse it and improve the stability, accuracy or
# efficiency of your setup modifying it.

# ### Visualize the solution

# In the previous part of the tutorial, we calculated the final solution of the given problem, now we want
# to visualize it. A more detailed explanation of visualization methods can be found in the section
# [Visualization](@ref visualization).

# #### Using Plots.jl

# The first option is to use the [Plots.jl](https://github.com/JuliaPlots/Plots.jl) package
# directly after the calculation, when the solution is saved in the `sol` variable. We connect the
# package and use the `plot` function.

using Plots
plot(sol)

# To show the mesh on the plot, we need to extract the visualization data from the solution as
# a [`PlotData2D`](@ref) object. Mesh extraction is possible using the [`getmesh`](@ref) function.
# Plots.jl has the `plot!` function that allows you to modify an already built graph.

pd = PlotData2D(sol)
plot!(getmesh(pd))

# #### Using Trixi2Vtk.jl

# Another way to visualize a solution is to extract it from a saved HDF5 file. After we used the
# `solve` function there is a file with the final solution. It is located in the `out` folder and
# is named as follows: `solution_index.h5`. The `index` is the final time step of the solution
# that is padded to 6 digits with zeros from the beginning. With [Trixi2Vtk](@ref) you
# can convert the HDF5 output file generated by Trixi.jl into a VTK file. This can be used in
# visualization tools such as [ParaView](https://www.paraview.org) or
# [VisIt](https://visit.llnl.gov) to plot the solution. The important thing is that currently
# Trixi2Vtk.jl supports conversion only for solutions in 2D and 3D spatial domains.

# If you haven't added a Trixi2Vtk.jl to your project yet, you can add it as follows.
# ```julia
# import Pkg
# Pkg.add(["Trixi2Vtk"])
# ```
# Now we are connecting the Trixi2Vtk.jl package and convert the file `out/solution_000018.h5` with
# the final solution using the [`trixi2vtk`](@ref) function saving the resulted file in the
# `out` folder.

using Trixi2Vtk
trixi2vtk(joinpath("out", "solution_000018.h5"), output_directory="out")

# Now two files `solution_000018.vtu` and `solution_000018_celldata.vtu` have been generated in the
# `out` folder. The first one contains all the information for visualizing the solution, the
# second one contains all the cell-based or discretization-based information.

# Now let's visualize the solution from the generated files in ParaView. Follow this short
# instruction to get the visualization.
# - Download, install and open [ParaView](https://www.paraview.org/download/)
# - Press `Ctrl+O` and browse the generated files `solution_000018.vtu` and
#   `solution_000018_celldata.vtu` from the `out` folder.
# - In the upper-left corner in the Pipeline Browser window, left-click on the eye-icon near
#   `solution_000018.vtu`.
# - In the lower-left corner in the Properties window, change the Coloring from Solid Color to
#   scalar. Now final solution visualization is already generated.
# - Now let's add the mesh to the visualization. In the upper-left corner in the
#   Pipeline Browser window, left-click on the eye-icon near `solution_000018_celldata.vtu`.
# - In the lower-left corner in the Properties window, change the Representation from the Surface
#   to the Wireframe. Then a white grid should appear on the visualization.
# Now, if you followed the instructions exactly, you should get an analog image, as shown in the
# section [Using Plots.jl](@ref Using-Plots.jl).

# ## Next steps: changing Trixi.jl itself

# If you plan on editing Trixi.jl itself, you can download Trixi.jl locally and run it from
# the cloned directory:

# ### Cloning Trixi.jl

# #### Windows

# If you are using Windows, you can clone Trixi.jl by using the GitHub Desktop tool:
# - If you do not have a GitHub account yet, create it on
#   the [GitHub website](https://github.com/join).
# - Download and install [GitHub Desktop](https://desktop.github.com/) and then log in into
#   your account.
# - Open GitHub Desktop, press `Ctrl+Shift+O`.
# - In the opened window, paste `trixi-framework/Trixi.jl` and choose the path to the folder where you want
#   to save Trixi.jl. Then click `Clone` and Trixi.jl will be cloned to your computer. 

# Now you cloned Trixi.jl and only need to tell Julia to use the local clone as the package sources:
# - Open a terminal using `Win+R` and `cmd`. Navigate to the folder with cloned Trixi.jl using `cd`.
# - Create new directory `run`, enter it, and start Julia with the `--project=.` flag:
#   ```shell
#   mkdir run 
#   cd run
#   julia --project=.
#   ```
# - Now run the following commands to install all relevant packages:
#   ```julia
#   using Pkg; Pkg.develop(PackageSpec(path="..")) # Install local Trixi.jl clone
#   Pkg.add(["OrdinaryDiffEq", "Plots"])  # Install additional packages
#   ```

# Now you already installed Trixi.jl from your local clone. Note that if you installed Trixi.jl this
# way, you always have to start Julia with the `--project` flag set to your `run` directory, e.g.,
# ```shell
# julia --project=.
# ```
# if already inside the `run` directory.

# #### Linux

# You can clone Trixi.jl to your computer executing following commands:
# ```shell
# git clone git@github.com:trixi-framework/Trixi.jl.git 
# # In case of an error, try the following:
# # git clone https://github.com/trixi-framework/Trixi.jl
# cd Trixi.jl
# mkdir run 
# cd run
# julia --project=. -e 'using Pkg; Pkg.develop(PackageSpec(path=".."))' # Install local Trixi.jl clone
# julia -e 'using Pkg; Pkg.add(["OrdinaryDiffEq", "Plots"])' # Install additional packages'
# ```
# Note that if you installed Trixi.jl this way,
# you always have to start Julia with the `--project` flag set to your `run` directory, e.g.,
# ```shell
# julia --project=.
# ```
# if already inside the `run` directory.

# ### For further reading

# To further delve into Trixi.jl, you may have a look at following tutorials.
# - [Introduction to DG methods](@ref scalar_linear_advection_1d) will teach you how to set up a simple way to
#   approximate the solution of a hyperbolic partial differential equation. It will be especially
#   useful to learn about the 
#   [Discontinuous Galerkin method](https://en.wikipedia.org/wiki/Discontinuous_Galerkin_method)
#   and the way it is implemented in Trixi.jl. Detailed explanations of the code provide a quick start
#   with Trixi.jl.
# - [Adding a new scalar conservation law](@ref adding_new_scalar_equations) and
#   [Adding a non-conservative equation](@ref adding_nonconservative_equation)
#   describe how to add new physics models that are not yet included in Trixi.jl.
# - [Callbacks](@ref callbacks-id) gives an overview of how to regularly execute specific actions
#   during a simulation, e.g., to store the solution or the adapt the mesh.

Sys.rm("out"; recursive=true, force=true) #hide #md
