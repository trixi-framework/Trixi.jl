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

# Trixi.jl works with the current stable Julia release. More information about Julia support can be
# found in the [`README.md`](https://github.com/trixi-framework/Trixi.jl#installation) file.
# A detailed description of the installation process can be found in the
# [Julia installation instructions](https://julialang.org/downloads/platform/).
# But you can follow also our short installation guidelines for Windows and Linux below.


# ### Windows

# - Download Julia installer for Windows from [https://julialang.org/downloads/](https://julialang.org/downloads/).
#   Make sure that you choose the right version of the installer (64-bit or 32-bit) according to
#   your computer.
# - Open the downloaded installer.
# - Paste an installation directory path or find it using a file manager (select `Browse`).
# - Select `Next`.
# - Check the `Add Julia to PATH` option to add Julia to the environment variables. 
#   This makes it possible to run Julia in the terminal from any directory by only typing `julia`.
# - Select `Next`, then Julia will be installed.

# Now you can verify, that Julia is installed:
# - Press `Win+R` on a keyboard.
# - Enter `cmd` in the opened window.
# - Enter in a terminal `julia`. 

# Then Julia will be started. To close Julia enter `exit()` or press `Ctrl+d`. 


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
# Then Julia will be started. To close Julia, enter `exit()` or press `Ctrl+d`.

# Note that later on in the tutorial, Julia will be started by only typing `julia` in a terminal.
# To enable that, you have to add
# [Julia to the PATH](https://julialang.org/downloads/platform/#linux_and_freebsd).


# ## Trixi.jl installation

# Trixi.jl and its related tools are registered Julia packages, thus their installation
# happens inside Julia.
# For a smooth workflow experience with Trixi.jl, you need to install 
# [Trixi.jl](https://github.com/trixi-framework/Trixi.jl),
# [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) and 
# [Plots.jl](https://github.com/JuliaPlots/Plots.jl).

# - Open a terminal and start Julia (Windows and Linux: type `julia`).
# - Execute following commands:
#   ```julia
#   import Pkg
#   Pkg.add(["OrdinaryDiffEq", "Plots", "Trixi"])
#   ```

# Now you have installed all these 
# packages. [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) provides time
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
# with a weak blast wave as the initial condition.

# Start Julia in a terminal and execute the following code:

# ```julia
# using Trixi, OrdinaryDiffEq
# trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"))
# ```
using Trixi, OrdinaryDiffEq #hide #md
trixi_include(@__MODULE__,joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl")) #hide #md

# To analyze the result of the computation, we can use the Plots.jl package and the function 
# `plot(...)`, which creates a graphical representation of the solution. `sol` is a variable
# defined in the executed example and it contains the solution at the final moment of the simulation.

using Plots
plot(sol)

# To obtain a list of all Trixi.jl elixirs execute
# [`get_examples`](@ref). It returns the paths to all example setups.

get_examples()

# Editing an existing elixir is the best way to start your first own investigation using Trixi.jl.


# ### Getting an existing setup file

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
# `elixir_euler_ec.jl`. In this example we consider the compressible Euler equations in two spatial
# dimensions,
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
# Here, ``\rho`` is the density, ``v_1`` and ``v_2`` are the velocities, ``e`` is the specific
# total energy, and
# ```math
# p = (\gamma - 1) \left( \rho e - \frac{1}{2} \rho (v_1^2+v_2^2) \right)
# ```
# is the pressure.
# Initial conditions consist of initial values for ``\rho``, ``\rho v_1``,
# ``\rho v_2`` and ``\rho e``.
# One of the common initial conditions for the compressible Euler equations is a simple density
# wave. Let's implement it.

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
# Then you will obtain a new solution from running the simulation with a different initial
# condition.

trixi_include(@__MODULE__,joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"), #hide #md
 initial_condition=initial_condition) #hide #md
pd = PlotData2D(sol) #hide #md
p1 = plot(pd["rho"]) #hide #md
p2 = plot(pd["v1"], clim=(0.05, 0.15)) #hide #md
p3 = plot(pd["v2"], clim=(0.15, 0.25)) #hide #md
p4 = plot(pd["p"], clim=(10, 30)) #hide #md
plot(p1, p2, p3, p4) #hide #md

# Feel free to make further changes to the initial condition to observe different solutions.

# Now you are able to download, modify and execute simulation setups for Trixi.jl. To explore
# further details on setting up a new simulation with Trixi.jl, refer to the second part of
# the introduction titled [Create first setup](@ref create_first_setup).

Sys.rm("out"; recursive=true, force=true) #hide #md