#src # Getting started

# Trixi.jl is a numerical simulation framework for conservation laws and
# is written in the [Julia programming language](https://julialang.org/).
# This tutorial is intended for beginners in Julia and Trixi.jl.
# After reading it, you will know how to install Julia and Trixi.jl on your computer,
# and you will be able to download setup files from our GitHub repository, modify them,
# and run simulations.

# The contents of this tutorial:
# - [Julia installation](@ref Julia-installation)
# - [Trixi.jl installation](@ref Trixi.jl-installation)
# - [Running a simulation](@ref Running-a-simulation)
# - [Getting an existing setup file](@ref Getting-an-existing-setup-file)
# - [Modifying an existing setup](@ref Modifying-an-existing-setup)

# ## Julia installation

# Trixi.jl is compatible with the latest stable release of Julia. Additional details regarding Julia
# support can be found in the [`README.md`](https://github.com/trixi-framework/Trixi.jl#installation)
# file. After installation, the current default Julia version can be managed through the command
# line tool `juliaup`. You may follow our concise installation guidelines for Windows, Linux, and
# MacOS provided below. In the event of any issues during the installation process, please consult
# the official [Julia installation instruction](https://julialang.org/downloads/).

# ### Windows

# - Open a terminal by pressing `Win+r` and entering `cmd` in the opened window.
# - To install Julia, execute the following command in the terminal:
#   ```shell
#   winget install julia -s msstore
#   ```
#   Note: For this installation an MS Store account is necessary to proceed.
# - Verify the successful installation of Julia by executing the following command in the terminal:
#   ```shell
#   julia
#   ```
#   To exit Julia, execute `exit()` or press `Ctrl+d`.

# ### Linux and MacOS

# - To install Julia, run the following command in a terminal:
#   ```shell
#   curl -fsSL https://install.julialang.org | sh
#   ```
#   Follow the instructions displayed in the terminal during the installation process.
# - If an error occurs during the execution of the previous command, you may need to install
#   `curl`. On Ubuntu-type systems, you can use the following command:
#   ```shell
#   sudo apt install curl
#   ```
#   After installing `curl`, repeat the first step once more to proceed with Julia installation.
# - Verify the successful installation of Julia by executing the following command in the terminal:
#   ```shell
#   julia
#   ```
#   To exit Julia, execute `exit()` or press `Ctrl+d`.

# ## Trixi.jl installation

# Trixi.jl and its related tools are registered Julia packages, thus their installation
# happens inside Julia.
# For a smooth workflow experience with Trixi.jl, you need to install 
# [Trixi.jl](https://github.com/trixi-framework/Trixi.jl),
# [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl), and 
# [Plots.jl](https://github.com/JuliaPlots/Plots.jl).

# - Open a terminal and start Julia.
# - Execute the following commands to install all mentioned packages. Please note that the
#   installation process involves downloading and precompiling the source code, which may take
#   some time depending on your machine.
#   ```julia
#   import Pkg
#   Pkg.add(["OrdinaryDiffEq", "Plots", "Trixi"])
#   ```
# - On Windows, the firewall may request permission to install packages.

# Besides Trixi.jl you have now installed two additional 
# packages: [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) provides time
# integration schemes used by Trixi.jl and [Plots.jl](https://github.com/JuliaPlots/Plots.jl)
# can be used to directly visualize Trixi.jl results from the Julia REPL.

# ## Usage

# ### Running a simulation

# To get you started, Trixi.jl has a large set
# of [example setups](https://github.com/trixi-framework/Trixi.jl/tree/main/examples), that can be
# taken as a basis for your future investigations. In Trixi.jl, we call these setup files
# "elixirs", since they contain Julia code that takes parts of Trixi.jl and combines them into
# something new.

# Any of the examples can be executed using the [`trixi_include`](@ref)
# function. `trixi_include(...)` expects
# a single string argument with a path to a file containing Julia code.
# For convenience, the [`examples_dir`](@ref) function returns a path to the
# [`examples`](https://github.com/trixi-framework/Trixi.jl/tree/main/examples)
# folder, which has been locally downloaded while installing Trixi.jl.
# `joinpath(...)` can be used to join path components into a full path. 

# Let's execute a short two-dimensional problem setup. It approximates the solution of
# the compressible Euler equations in 2D for an ideal gas ([`CompressibleEulerEquations2D`](@ref))
# with a weak blast wave as the initial condition and periodic boundary conditions. 

# The compressible Euler equations in two spatial dimensions are given by
# ```math
# \frac{\partial}{\partial t}
# \begin{pmatrix}
# \rho \\ \rho v_1 \\ \rho v_2 \\ \rho e
# \end{pmatrix}
# +
# \frac{\partial}{\partial x}
# \begin{pmatrix}
# \rho v_1 \\ \rho v_1^2 + p \\ \rho v_1 v_2 \\ (\rho e + p) v_1
# \end{pmatrix}
# +
# \frac{\partial}{\partial y}
# \begin{pmatrix}
# \rho v_2 \\ \rho v_1 v_2 \\ \rho v_2^2 + p \\ (\rho e + p) v_2
# \end{pmatrix}
# =
# \begin{pmatrix}
# 0 \\ 0 \\ 0 \\ 0
# \end{pmatrix},
# ```
# for an ideal gas with the specific heat ratio ``\gamma``.
# Here, ``\rho`` is the density, ``v_1`` and ``v_2`` are the velocities, ``e`` is the specific
# total energy, and
# ```math
# p = (\gamma - 1) \left( \rho e - \frac{1}{2} \rho (v_1^2 + v_2^2) \right)
# ```
# is the pressure.

# The [`initial_condition_weak_blast_wave`](@ref) is specified in
# [`compressible_euler_2d.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/src/equations/compressible_euler_2d.jl) 

# Start Julia in a terminal and execute the following code:

# ```julia
# using Trixi, OrdinaryDiffEq
# trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"))
# ```
using Trixi, OrdinaryDiffEq #hide #md
trixi_include(@__MODULE__, joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl")) #hide #md

# The output contains a recap of the setup and various information about the course of the simulation.
# For instance, the solution was approximated over the [`TreeMesh`](@ref) with 1024 effective cells using
# the `CarpenterKennedy2N54` ODE
# solver. Further details about the ODE solver can be found in the
# [documentation of OrdinaryDiffEq.jl](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Low-Storage-Methods)

# To analyze the result of the computation, we can use the Plots.jl package and the function 
# `plot(...)`, which creates a graphical representation of the solution. `sol` is a variable
# defined in the executed example and it contains the solution after the simulation 
# finishes. `sol.u` holds the vector of values at each saved timestep, while `sol.t` holds the
# corresponding times for each saved timestep. In this instance, only two timesteps were saved: the
# initial and final ones. The plot depicts the distribution of the weak blast wave at the final moment
# of time, showing the density, velocities, and pressure of the ideal gas across a 2D domain.

using Plots
plot(sol)

# ### Getting an existing setup file

# To obtain a list of all Trixi.jl elixirs execute
# [`get_examples`](@ref). It returns the paths to all example setups.

get_examples()

# Editing an existing elixir is the best way to start your first own investigation using Trixi.jl.

# To edit an existing elixir, you first have to find a suitable one and then copy it to a local
# folder. Let's have a look at how to download the `elixir_euler_ec.jl` elixir used in the previous
# section from the [Trixi.jl GitHub repository](https://github.com/trixi-framework/Trixi.jl).

# - All examples are located inside
#   the [`examples`](https://github.com/trixi-framework/Trixi.jl/tree/main/examples) folder.
# - Navigate to the
#   file [`elixir_euler_ec.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_euler_ec.jl).
# - Right-click the `Raw` button on the right side of the webpage and choose `Save as...`
#   (or `Save Link As...`).
# - Choose a folder and save the file.

# ### Modifying an existing setup

# As an example, we will change the initial condition for calculations that occur in
# `elixir_euler_ec.jl`. Initial conditions for [`CompressibleEulerEquations2D`](@ref) consist of
# initial values for ``\rho``, ``\rho v_1``, ``\rho v_2`` and ``\rho e``. One of the common initial
# conditions for the compressible Euler equations is a simple density wave. Let's implement it.

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
    rho = 1.0 + 0.98 * sinpi(sum(x) - t * (v1 + v2)) # density wave profile
    p = 20 # pressure
    rho_e = p / (equations.gamma - 1) + 1 / 2 * rho * (v1^2 + v2^2)
    return SVector(rho, rho * v1, rho * v2, rho_e)
end
initial_condition = initial_condition_density_waves
nothing; #hide #md

# - Execute the following code one more time, but instead of `path/to/file` paste the path to the
#   `elixir_euler_ec.jl` file that you just edited.
#   ```julia
#   using Trixi
#   trixi_include(path/to/file)
#   using Plots
#   plot(sol)
#   ```
# Then you will obtain a new solution from running the simulation with a different initial
# condition.

trixi_include(@__MODULE__, joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"), #hide #md
              initial_condition = initial_condition) #hide #md
pd = PlotData2D(sol) #hide #md
p1 = plot(pd["rho"]) #hide #md
p2 = plot(pd["v1"], clim = (0.05, 0.15)) #hide #md
p3 = plot(pd["v2"], clim = (0.15, 0.25)) #hide #md
p4 = plot(pd["p"], clim = (10, 30)) #hide #md
plot(p1, p2, p3, p4) #hide #md

# To get exactly the same picture execute the following.
# ```julia
# pd = PlotData2D(sol)
# p1 = plot(pd["rho"])
# p2 = plot(pd["v1"], clim=(0.05, 0.15))
# p3 = plot(pd["v2"], clim=(0.15, 0.25))
# p4 = plot(pd["p"], clim=(10, 30))
# plot(p1, p2, p3, p4)
# ```

# Feel free to make further changes to the initial condition to observe different solutions.

# Now you are able to download, modify and execute simulation setups for Trixi.jl. To explore
# further details on setting up a new simulation with Trixi.jl, refer to the second part of
# the introduction titled [Create your first setup](@ref create_first_setup).

Sys.rm("out"; recursive = true, force = true) #hide #md
