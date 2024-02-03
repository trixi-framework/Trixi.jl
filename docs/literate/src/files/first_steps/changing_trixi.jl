#src # Changing Trixi.jl itself

# If you plan on editing Trixi.jl itself, you can download Trixi.jl locally and run it from
# the cloned directory.


# ## Cloning Trixi.jl


# ### Windows

# If you are using Windows, you can clone Trixi.jl by using the GitHub Desktop tool:
# - If you do not have a GitHub account yet, create it on
#   the [GitHub website](https://github.com/join).
# - Download and install [GitHub Desktop](https://desktop.github.com/) and then log in to
#   your account.
# - Open GitHub Desktop, press `Ctrl+Shift+O`.
# - In the opened window, paste `trixi-framework/Trixi.jl` and choose the path to the folder where
#   you want to save Trixi.jl. Then click `Clone` and Trixi.jl will be cloned to your computer. 

# Now you cloned Trixi.jl and only need to tell Julia to use the local clone as the package sources:
# - Open a terminal using `Win+R` and `cmd`. Navigate to the folder with the cloned Trixi.jl using `cd`.
# - Create a new directory `run`, enter it, and start Julia with the `--project=.` flag:
#   ```shell
#   mkdir run 
#   cd run
#   julia --project=.
#   ```
# - Now run the following commands to install all relevant packages:
#   ```julia
#   julia> using Pkg; Pkg.develop(PackageSpec(path="..")) # Tell Julia to use the local Trixi.jl clone
#   julia> Pkg.add(["OrdinaryDiffEq", "Plots"])  # Install additional packages
#   ```

# Now you already installed Trixi.jl from your local clone. Note that if you installed Trixi.jl
# this way, you always have to start Julia with the `--project` flag set to your `run` directory,
# e.g.,
# ```shell
# julia --project=.
# ```
# if already inside the `run` directory.


# ### Linux

# You can clone Trixi.jl to your computer by executing the following commands:
# ```shell
# git clone git@github.com:trixi-framework/Trixi.jl.git 
# # If an error occurs, try the following:
# # git clone https://github.com/trixi-framework/Trixi.jl
# cd Trixi.jl
# mkdir run 
# cd run
# julia --project=. -e 'using Pkg; Pkg.develop(PackageSpec(path=".."))' # Tell Julia to use the local Trixi.jl clone
# julia --project=. -e 'using Pkg; Pkg.add(["OrdinaryDiffEq", "Plots"])' # Install additional packages'
# ```
# Note that if you installed Trixi.jl this way,
# you always have to start Julia with the `--project` flag set to your `run` directory, e.g.,
# ```shell
# julia --project=.
# ```
# if already inside the `run` directory.


# ## For further reading

# To further delve into Trixi.jl, you may have a look at following tutorials.
# - [Introduction to DG methods](@ref scalar_linear_advection_1d) will teach you how to set up a
#   simple way to approximate the solution of a hyperbolic partial differential equation. It will
#   be especially useful to learn about the 
#   [Discontinuous Galerkin method](https://en.wikipedia.org/wiki/Discontinuous_Galerkin_method)
#   and the way it is implemented in Trixi.jl.
# - [Adding a new scalar conservation law](@ref adding_new_scalar_equations) and
#   [Adding a non-conservative equation](@ref adding_nonconservative_equation)
#   describe how to add new physics models that are not yet included in Trixi.jl.
# - [Callbacks](@ref callbacks-id) gives an overview of how to regularly execute specific actions
#   during a simulation, e.g., to store the solution or adapt the mesh.
