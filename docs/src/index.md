# Trixi: A tree-based flexible DG/SBP framework written in Julia

*Trixi* is a flexible DG/SBP framework written in the Julia programming
language. It is based on a two-dimensional hierarchical mesh (quadtree) and aims
to be easy to use and extend.


## Installation
Strictly speaking, no installation is necessary to run Trixi. However, the
simulation program and the postprocessing tools rely on a number of Julia
packages, which need to be available on the respective machine. This can most
easily be achieved by performing the following steps:

1.  Clone the repository:
    ```bash
    git clone git@gitlab.mi.uni-koeln.de:numsim/code/Trixi.jl.git
    ```
2.  Enter the cloned directory and run the following command to install all
    required dependencies:
    ```bash
    julia utils/install.jl
    ```

Afterwards you are able to use Trixi and the postprocessing tools without
repeating these steps. In case the execution of the `install.jl` script fails,
you can also install the dependencies manually:
```bash
# Enter the Trixi root directory
cd path/to/Trixi.jl

# Install Trixi dependencies
julia --project=. -e 'import Pkg; Pkg.instantiate()'

# Install Trixi2Img dependencies
julia --project='postprocessing/pkg/Trixi2Img' -e 'import Pkg; Pkg.instantiate()'

# Install Trixi2Vtk dependencies
julia --project='postprocessing/pkg/Trixi2Vtk' -e 'import Pkg; Pkg.instantiate()'
```


## Usage
Enter the root directory `Trixi.jl/` and run
```bash
bin/trixi parameters.toml
```

To change the simulation setup, edit `parameters.toml`. You can also pass a different
parameters file on the command line, e.g., `bin/trixi awesome_parameters.toml`.


## Authors
Trixi was created by Michael Schlottke-Lakemper and Gregor Gassner.
