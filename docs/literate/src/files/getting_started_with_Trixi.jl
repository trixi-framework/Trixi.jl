#src # Getting started with Trixi.jl

# This tutorial is intended for beginners in Julia and Trixi.jl.
# After reading it, you will install Julia and Trixi on your PC and will be able to download files
# from Trixi github, execute and edit them.
# **Trixi.jl** is a numerical simulation framework for hyperbolic conservation laws 
# written in [`Julia`](https://julialang.org/).
# This means that Julia have to be installed on a PC to work with Trixi. 

# ## Julia installation

# Trixi works with the current stable Julia release. More information about Julia support can be
# found in [`README`](https://github.com/trixi-framework/Trixi.jl#installation).
# The most fully explaind installation process can be found in
# this [`Julia installation instruction`](https://julialang.org/downloads/platform/).
# But you can follow also our short installation instruction.

# ### Windows

# - Download Julia [`installer`](https://julialang.org/downloads/) for Windows. Make sure 
#   that you chose the right version of installer (64-bit or 32-bit) according to your computer.
# - Open the downloaded installer.
# - Paste an installation directory path or find it using a file manager (select `Browse`).
# - Select `Next`.
# - Check the `Add Julia to PATH` to add Julia to Environment Variables. 
#   This makes possible to run Julia using Terminal from any directory only typing `julia`.
# - Select `Next`, then Julia will be insalled.

# Now you can verify, that Julia is installed:
# - Type `Win+R` on a keyboard.
# - Enter `cmd` in opened window.
# - Enter in a terminal `julia`. 

# Then Julia will be invoked. To close Julia enter `exit()`. 

# ### Linux

# - Open a terminal and navigate (using `cd`) to a directory, where you want to save Julia.
#   Or you can open file manager, find this directory, right-click inside and 
#   choose `Open Terminal Here`.
# - To install Julia execute the following commands in the Terminal:
#   ````
#   wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz
#   tar zxvf julia-1.8.5-linux-x86_64.tar.gz
#   ````
#   Now you can verify that Julia is installed entering `julia` command in the Terminal.

# Then Julia will be invoked. To close Julia enter `exit()`.

# ## Trixi installation

# Trixi and its related tools are registered Julia packages. So installation of them is
# running inside Julia. For appropriate work of Trixi you need to install 
# [`Trixi`](https://github.com/trixi-framework/Trixi.jl),
# [`Trixi2Vtk`](https://github.com/trixi-framework/Trixi2Vtk.jl),
# [`OrdinaryDiffEq`](https://github.com/SciML/OrdinaryDiffEq.jl) and 
# [`Plots`](https://github.com/JuliaPlots/Plots.jl).

# - Open a Terminal: type `Win + R` and enter `cmd`.
# - Invoke Julia executing `julia`.
# - Execute following commands:
#   ````
#   import Pkg
#   Pkg.add(["Trixi", "Trixi2Vtk", "OrdinaryDiffEq", "Plots"])
#   ````

# Now you have installed all this 
# packages. [`Trixi2Vtk`](https://github.com/trixi-framework/Trixi2Vtk.jl) is a visualization
# tool, [`OrdinaryDiffEq`](https://github.com/SciML/OrdinaryDiffEq.jl) provides time integration schemes
# used by Trixi and [`Plots`](https://github.com/JuliaPlots/Plots.jl) can be used to directly
# visualize Trixi's results from the Julia REPL.

# ## Usage

# ### Files execution

# Trixi has a big set
# of [`examples`](https://github.com/trixi-framework/Trixi.jl/tree/main/examples), that can be taken
# as basis for your future investigations.

# Now execute one of them using 
# [`trixi_include(...)`](https://trixi-framework.github.io/Trixi.jl/stable/reference-trixi/#Trixi.trixi_include-Tuple{Module,%20AbstractString})
# function. `trixi_include(...)` expects
# a single string argument with the path to a text file containing Julia code.
# `joinpath(...)` join a path components into a full path. `examples_dir()` returns a path to the
# [`examples`](https://github.com/trixi-framework/Trixi.jl/tree/main/examples) folder.

# Let's execute short two-dimensional problem setup. Which approximates solution of
# [`compressible Euler equations in 2D for an ideal gas`](https://trixi-framework.github.io/Trixi.jl/stable/reference-trixi/#Trixi.CompressibleEulerEquations2D)
# with
# [`weak blast wave initial condition`](https://trixi-framework.github.io/Trixi.jl/stable/reference-trixi/#Trixi.initial_condition_weak_blast_wave-Tuple{Any,%20Any,%20CompressibleEulerEquations2D})

# Invoke Julia in terminal. (Open Terminal: `Win+R` and enter `cmd`, invoke Julia in terminal, e.g.: 
# `julia --project=@.`).
# And execute following code. (*Remark:* you can ignore all arguments of trixi_include() except
# path to the file) 

using Trixi, OrdinaryDiffEq
trixi_include(@__MODULE__,joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_ec.jl"),
 callbacks=CallbackSet(StepsizeCallback(cfl=1.0)))

# To observe result of computation, we need to use `Plots` package and function `plot()`, that
# builds a graphical representation of the solution. `sol` is a variable defined in
# executed example and it contains a solution at the final moment of time.

using Plots
plot(sol)

# To obtain list of all Trixi elixirs execute `get_examples()`. This will
# return pathes to all examples.

get_examples()

# Editing the Trixi examples is the best way to start your first own investigation using Trixi.

# ### Files downloading

# To edit example files you have to download them. Let's have a look how to download
# `elixir_euler_ec.jl` used in previous section from
# [`Trixi github`](https://github.com/trixi-framework/Trixi.jl).

# - All examples are located inside
#   the [`examples`](https://github.com/trixi-framework/Trixi.jl/tree/main/examples) folder.
# - Navigate to the
#   file [`elixir_euler_ec.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_euler_ec.jl).
# - Click the `Raw` button on right side of the webpage.
# - Right-click on any place of newly opened webpage and choose `Save as`.
# - Choose folder and erase `.txt` from the file name. Save the file.

# ### Files editing

# For example, we will change the initial condition for calculations that occur in the
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
# So this means that initial condition consists of initial values for ``\rho``, ``\rho v_1``,
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
# - Now you can create your own initial conditions. Write following code into a file after
#   commented out line:

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
# - In opened window paste `trixi-framework/Trixi.jl` and choose path to a folder, where you want
#   to save Trixi. Then click `Clone` and Trixi will be cloned to PC. 

# Now you cloned Trixi and only need to add Trixi packages to Julia.
# - Open Terminal using `Win+R` and `cmd`. Navigate to the folder with cloned Trixi using `cd`.
# - Start Julia with the `--project` flag set to your local Trixi clone, e.g.,
#   ````
#   julia --project=@.
#   ````
# - Run following commands in Julia REPL:
#   ````
#   import Pkg; Pkg.instantiate()
#   Pkg.add(["Trixi2Vtk", "Plots", "OrdinaryDiffEq"])
#   ````

# Now you already installed Trixi from your local clone. Note that if you installed Trixi this way,
# you always have to start Julia with the `--project` flag set to your local Trixi clone, e.g.,
# ````
# julia --project=@.
# ````

# #### Linux

# You can clone Trixi to PC executing following commands:
# ````
# git clone git@github.com:trixi-framework/Trixi.jl.git 
# # In case of an error, use following: git clone https://github.com/trixi-framework/Trixi.jl
# cd Trixi.jl
# julia --project=@. -e 'import Pkg; Pkg.instantiate()'
# julia -e 'import Pkg; Pkg.add(["Trixi2Vtk", "Plots"])'
# julia -e 'import Pkg; Pkg.add("OrdinaryDiffEq")'
# ````
# Note that if you installed Trixi this way,
# you always have to start Julia with the `--project` flag set to your local Trixi clone, e.g.,
# ````
# julia --project=@.
# ````

# ### For further reading

# To get deeper into Trixi, you may have a look at following tutorials.
# - [`Introduction to DG methods`](https://trixi-framework.github.io/Trixi.jl/stable/tutorials/scalar_linear_advection_1d/)
#   is about how to set up a simple way to approximate the solution of a hyperbolic partial
#   differential equation. It will be esspecialy useful to learn about
#   [`Discontinuous Galerkin method`](https://en.wikipedia.org/wiki/Discontinuous_Galerkin_method)
#   and how it implemented in Trixi. Detailed explanation of code provides quick start with Trixi.
# - [`Adding a new scalar conservation law`](https://trixi-framework.github.io/Trixi.jl/stable/tutorials/adding_new_scalar_equations/)
#   and
#   [`Adding a non-conservative equation`](https://trixi-framework.github.io/Trixi.jl/stable/tutorials/adding_nonconservative_equation/)
#   describe how to add a new physics model that's not yet included in Trixi.jl.
# - [`Callbacks`](https://trixi-framework.github.io/Trixi.jl/stable/callbacks/)
#   gives an overview of an algorithmic entity called callback that gets passed to the ODE solver
#   and is called at specific points during execution to perform certain tasks.
#   It extends Trixi without modifying the internal source code.