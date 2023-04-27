#src # Getting started

# This tutorial is intended for beginners in Julia and Trixi.jl.
# After reading it, you will know how to install Julia and Trixi.jl on your computer,
# and you will be able to download setup files from our GitHub repository, modify them,
# and run simulations.
# **Trixi.jl** is a numerical simulation framework for hyperbolic conservation laws 
# written in [`Julia`](https://julialang.org/).

# ## Julia installation

# Trixi works with the current stable Julia release. More information about Julia support can be
# found in the [`README.md`](https://github.com/trixi-framework/Trixi.jl#installation) file.
# A detailed description of the installation process can be found in the
# [Julia installation instructions](https://julialang.org/downloads/platform/).
# But you can follow also our short installation guidelines below.

# ### Windows

# - Download Julia installer for Windows from https://julialang.org/downloads/. Make sure 
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
# - To install Julia execute the following commands in a Terminal:
#   ````
#   wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz
#   tar zxvf julia-1.8.5-linux-x86_64.tar.gz
#   ````
#   Now you can verify that Julia is installed entering `julia` command in a Terminal.

# Then Julia will be invoked. To close Julia, enter `exit()` or press `Ctrl+d`.

# ## Trixi installation

# Trixi and its related tools are registered Julia packages, thus their installation
# happens inside Julia.
# For a smooth workflow experience with Trixi.jl, you need to install 
# [Trixi.jl](https://github.com/trixi-framework/Trixi.jl),
# [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) and 
# [Plots.jl](https://github.com/JuliaPlots/Plots.jl).

# - Open a terminal.
# - Invoke Julia executing `julia`.
# - Execute following commands:
#   ````
#   import Pkg
#   Pkg.add(["OrdinaryDiffEq", "Plots", "Trixi"])
#   ````

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

# Invoke Julia in a Terminal.
# And execute following code.

using Trixi, OrdinaryDiffEq
trixi_include(@__MODULE__,joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"))

# To observe the result of the computation, we need to use the `Plots` package and the function 
# `plot()`, that builds a graphical representation of the solution. `sol` is a variable defined in
# executed example and it contains a solution at a final moment of time.

using Plots
plot(sol)

# To obtain list of all Trixi elixirs execute
# [`get_examples()`](https://trixi-framework.github.io/Trixi.jl/stable/reference-trixi/#Trixi.get_examples-Tuple{}).
# This will return paths to all examples.

get_examples()

# Editing Trixi examples is the best way to start your first own investigation using Trixi.

# ### Files downloading

# To edit example files you have to download them. Let's have a look how to download
# `elixir_euler_ec.jl` used in the previous section from the
# [`Trixi github`](https://github.com/trixi-framework/Trixi.jl).

# - All examples are located inside
#   the [`examples`](https://github.com/trixi-framework/Trixi.jl/tree/main/examples) folder.
# - Navigate to the
#   file [`elixir_euler_ec.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_euler_ec.jl).
# - Click the `Raw` button on the right side of the webpage.
# - Right-click on any place of the newly opened webpage and choose `Save as`.
# - Choose a folder and erase `.txt` from the file name. Save the file.

# ### Files editing

# For example, we will change the initial condition for calculations that occur in
# `elixir_euler_ec.jl`. In this example we consider the compressible Euler equations:
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
# \end{pmatrix}
# ```
# for an ideal gas with ratio of specific heats ``\gamma`` 
# in two space dimensions.
# Here, ``\rho`` is the density, ``v_1``, ``v_2`` the velocities, ``e`` the specific total
# energy, and
# ```math
# p = (\gamma - 1) \left( \rho e - \frac{1}{2} \rho (v_1^2+v_2^2) \right)
# ```
# the pressure.
# Initial condition consists of initial values for ``\rho``, ``\rho v_1``,
# ``\rho v_2`` and ``\rho e``.
# One of the common initial condition for compressible Euler equations is density wave.
# Let's implement it.

# - Open the downloaded file with notepad or any other text editor.
# - And go to the 9th line with following code:
#   ````
#   initial_condition = initial_condition_weak_blast_wave
#   ````
#   Here
#   [`initial_condition_weak_blast_wave`](https://trixi-framework.github.io/Trixi.jl/stable/reference-trixi/#Trixi.initial_condition_weak_blast_wave-Tuple{Any,%20Any,%20CompressibleEulerEquations2D})
#   is used.
# - Comment out this line using # symbol:
#   ````
#   # initial_condition = initial_condition_weak_blast_wave
#   ````
# - Now you can create your own initial conditions. Write following code into a file after the
#   commented line:

    function initial_condition_density_waves(x,t,equations::CompressibleEulerEquations2D)
      v1 = 0.1 # velocity along x-axis
      v2 = 0.2 # velocity along y-axis
      rho = 1.0 + 0.98 * sin(pi * (sum(x) - t * (v1 + v2))) # density wave profile
      p = 20 # pressure
      rho_e = p / (equations.gamma - 1) + 1/2 * rho * (v1^2 + v2^2)
      return SVector(rho, rho*v1, rho*v2, rho_e)
    end
    initial_condition=initial_condition_density_waves

# - Execute following code one more time, but instead of `path_to_file` paste the path to the
#   `elixir_euler_ec.jl` file from the current folder.
#   ````
#   using Trixi
#   trixi_include(path_to_file)
#   using Plots
#   plot(sol)
#   ````
# Then you will obtain new solution.

trixi_include(@__MODULE__,joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"), #hide #md
  callbacks=CallbackSet(StepsizeCallback(cfl=1.0)), initial_condition=initial_condition) #hide #md
plot(sol) #hide #md

# Feel free to add
# changes into `initial_condition` to observe different solutions.

# Now you are able to download, edit and execute Trixi code.

# ## Next steps

# If you plan on editing Trixi itself, you can download Trixi locally and run it from
# the cloned directory:

# ### Cloning Trixi

# #### Windows

# If you are using Windows OS, you can clone Trixi directory using a Github Desktop.
# - If you haven't any github account yet, you have to create it on
#   the [`Github website`](https://github.com/join).
# - Download and install [`Github Desktop`](https://desktop.github.com/) and then login into
#   your account.
# - Open an installed Github Desktop, type `Ctrl+Shift+O`.
# - In opened window paste `trixi-framework/Trixi.jl` and choose path to the folder, where you want
#   to save Trixi. Then click `Clone` and Trixi will be cloned to PC. 

# Now you cloned Trixi and only need to add Trixi packages to Julia.
# - Open a Terminal using `Win+R` and `cmd`. Navigate to the folder with cloned Trixi using `cd`.
# - Create new directory and start Julia with the `--project` flag set to your local Trixi clone.
#   ````
#   mkdir run 
#   cd run
#   julia --project=.
#   ````
# - Run following commands in Julia REPL:
#   ````
#   using Pkg; Pkg.develop(PackageSpec(path="..")) # Install local Trixi clone
#   Pkg.add(["OrdinaryDiffEq", "Trixi2Vtk", "Plots"])  # Install additional packages
#   ````

# Now you already installed Trixi from your local clone. Note that if you installed Trixi this way,
# you always have to start Julia with the `--project` flag set to your `run` directory, e.g.,
# ````
# julia --project=.
# ````
# if already inside the `run` directory.

# #### Linux

# You can clone Trixi to PC executing following commands:
# ````
# git clone git@github.com:trixi-framework/Trixi.jl.git 
# # In case of an error, use following: git clone https://github.com/trixi-framework/Trixi.jl
# cd Trixi.jl
# mkdir run 
# cd run
# julia --project=. -e 'using Pkg; Pkg.develop(PackageSpec(path=".."))' # Install local Trixi clone
# julia -e 'using Pkg; Pkg.add(["OrdinaryDiffEq", "Trixi2Vtk", "Plots"])' # Install additional packages'
# ````
# Note that if you installed Trixi this way,
# you always have to start Julia with the `--project` flag set to your `run` directory, e.g.,
# ````
# julia --project=.
# ````
# if already inside the `run` directory.

# ### For further reading

# To get deeper into Trixi, you may have a look at following tutorials.
# - [`Introduction to DG methods`](https://trixi-framework.github.io/Trixi.jl/stable/tutorials/scalar_linear_advection_1d/)
#   is about how to set up a simple way to approximate the solution of a hyperbolic partial
#   differential equation. It will be esspecialy useful to learn about the
#   [`Discontinuous Galerkin method`](https://en.wikipedia.org/wiki/Discontinuous_Galerkin_method)
#   and way of its implementation in Trixi. Detailed explanation of code provides a quick start
#   with Trixi.
# - [`Adding a new scalar conservation law`](https://trixi-framework.github.io/Trixi.jl/stable/tutorials/adding_new_scalar_equations/)
#   and
#   [`Adding a non-conservative equation`](https://trixi-framework.github.io/Trixi.jl/stable/tutorials/adding_nonconservative_equation/)
#   describe how to add a new physics model that's not included in Trixi.jl yet.
# - [`Callbacks`](https://trixi-framework.github.io/Trixi.jl/stable/callbacks/)
#   gives an overview of an algorithmic entity called callback that gets passed to the ODE solver
#   and is called at specific points during execution to perform certain tasks.
#   It extends Trixi without modifying the internal source code.

Sys.rm("out"; recursive=true, force=true) #hide #md