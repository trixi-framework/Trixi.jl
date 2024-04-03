#src # Changing Trixi.jl itself

# If you plan on editing Trixi.jl itself, you can download Trixi.jl locally and run it from
# the cloned directory.


# ## Forking Trixi.jl

# To create your own fork of Trixi.jl, log in to your GitHub account, visit the
# [Trixi.jl GitHub repository](https://github.com/trixi-framework/Trixi.jl) and click the `Fork`
# button located in the upper-right corner of the page. Then, click on `Create fork` in the opened
# window to complete the forking process.


# ## Cloning Trixi.jl


# ### Windows

# If you are using Windows, you can clone Trixi.jl by using the GitHub Desktop tool:
# - If you do not have a GitHub account yet, create it on
#   the [GitHub website](https://github.com/join).
# - Download and install [GitHub Desktop](https://desktop.github.com/) and then log in to
#   your account.
# - Open GitHub Desktop, press `Ctrl+Shift+O`.
# - In the opened window, navigate to the `URL` tab and paste `trixi-framework/Trixi.jl` or
#   `YourGitHubUserName/Trixi.jl` to clone your own fork of Trixi.jl, and choose the
#   path to the folder where you want to save Trixi.jl. Then click `Clone` and Trixi.jl will be
#   cloned to your computer. 

# Now you cloned Trixi.jl and only need to tell Julia to use the local clone as the package sources:
# - Open a terminal using `Win+r` and `cmd`. Navigate to the folder with the cloned Trixi.jl using `cd`.
# - Create a new directory `run`, enter it, and start Julia with the `--project=.` flag:
#   ```shell
#   mkdir run 
#   cd run
#   julia --project=.
#   ```
# - Now run the following commands to install all relevant packages:
#   ```julia
#   using Pkg; Pkg.develop(PackageSpec(path="..")) # Tell Julia to use the local Trixi.jl clone
#   Pkg.add(["OrdinaryDiffEq", "Plots"])  # Install additional packages
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
# julia --project=. -e 'using Pkg; Pkg.add(["OrdinaryDiffEq", "Plots"])' # Install additional packages
# ```
# Alternatively, you can clone your own fork of Trixi.jl by replacing the link
# `git@github.com:trixi-framework/Trixi.jl.git` with `git@github.com:YourGitHubUserName/Trixi.jl.git`.

# Note that if you installed Trixi.jl this way,
# you always have to start Julia with the `--project` flag set to your `run` directory, e.g.,
# ```shell
# julia --project=.
# ```
# if already inside the `run` directory.


# ## Developing Trixi.jl

# If you've created and cloned your own fork of Trixi.jl, you can make local changes to Trixi.jl
# and propose them as a Pull Request (PR) to be merged into `trixi-framework/Trixi.jl`.

# Linux and MacOS utilize the `git` version control system to manage changes between your local and
# remote repositories. The most commonly used commands include `add`, `commit`, `push` and `pull`.
# You can find detailed information about these functions in the
# [Git documentation](https://git-scm.com/docs).

# For Windows and GitHub Desktop users, refer to the
# [documentation of GitHub Desktop](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop#making-changes-in-a-branch).

# After making local changes to Trixi.jl and pushing them to the remote repository, you can open a
# Pull Request (PR) from your branch to the main branch of `trixi-framework/Trixi.jl`. Then, follow
# the Review checklist provided in the Pull Request to streamline the review process.


# ## Additional reading

# To further delve into Trixi.jl, you may have a look at the following introductory tutorials.
# - [Behind the scenes of a simulation setup](@ref behind_the_scenes_simulation_setup) will guide
#   you through a simple Trixi.jl setup ("elixir"), giving an overview of what happens in the
#   background during the initialization of a simulation. It clarifies some of the more
#   fundamental, technical concepts that are applicable to a variety of (also more complex)
#   configurations.
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
