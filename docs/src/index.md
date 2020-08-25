# Trixi

*Trixi* is a flexible DG/SBP framework written in the Julia programming
language. It is based on a two-dimensional hierarchical mesh (quadtree) and aims
to be easy to use and extend.


## Installation
If you have not yet installed Julia, please follow the instructions for your
operating system found [here](https://julialang.org/downloads/platform/). Trixi
works with Julia v1.3 or higher.
Official binaries are available for Windows, macOS, Linux, and FreeBSD.

Strictly speaking, no installation is necessary to run Trixi. However, the
simulation program and the postprocessing tools rely on a number of Julia
packages, which need to be available on the respective machine. This can most
easily be achieved by performing the following steps:

  1. Clone the repository:
     ```
     git clone git@github.com:trixi-framework/Trixi.jl.git
     ```
  2. Enter the cloned directory and run the following command to install all
     required dependencies:
     ```
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
The `install.jl` script can also be used to *update* the dependencies if they have
changed since you installed Trixi.

### Example: Installing Trixi from scratch
```@raw html
  <script id="asciicast-pGwc6GpZ5AFlb8Dk9gsACMR24"
          src="https://asciinema.org/a/pGwc6GpZ5AFlb8Dk9gsACMR24.js"
          async
          data-cols=90
          data-rows=20
          data-speed=4></script>
```
Please note that the playback speed is set to 4x, thus the entire installation
procedure lasts around 4 minutes in real time (depending on the performance of
your computer and on how many dependencies had already been installed before).


## Usage
Enter the root directory `Trixi.jl/` and execute
```bash
bin/trixi
```
This will start an interactive Julia session with the Trixi module already
loaded. To run a simulation, execute
```julia
Trixi.run("examples/parameters.toml")
```
You can also pass a different parameters file or edit `examples/parameters.toml` to
modify the simulation setup.
More information on how to use Trixi interactively can be found in the
[Development](@ref) section.

Sometimes it can be helpful to run Trixi non-interactively in batch mode, e.g., when starting
a simulation from another script. This is possible by directly passing the
parameters file to Trixi on the command line:
```bash
bin/trixi examples/parameters.toml
```

### Example: Running Trixi interactively
```@raw html
  <script id="asciicast-zn79qrdAfCDGWKlQgWHzc0wCB"
          src="https://asciinema.org/a/zn79qrdAfCDGWKlQgWHzc0wCB.js"
          async
          data-cols=90
          data-rows=48></script>
```

### Performing a convergence analysis
To automatically determine the experimental order of convergence (EOC) for a
given setup, execute
```julia
Trixi.convtest("examples/parameters.toml", 4)
```
This will run a convergence test with the parameters file `examples/parameters.toml`,
using four iterations with different initial refinement levels. The initial
iteration will use the parameters file unchanged, while for each subsequent
iteration the `initial_refinement_level` parameter is incremented by one.
Finally, the measured ``L^2`` and ``L^\infty`` errors and the determined EOCs
will be displayed like this:
```
[...]
L2
scalar
error     EOC
9.14e-06  -
5.69e-07  4.01
3.55e-08  4.00
2.22e-09  4.00

mean      4.00
--------------------------------------------------------------------------------
Linf
scalar
error     EOC
6.44e-05  -
4.11e-06  3.97
2.58e-07  3.99
1.62e-08  4.00

mean      3.99
--------------------------------------------------------------------------------
```

An example with multiple variables looks like this:
```julia
julia> Trixi.convtest("examples/parameters_source_terms.toml", 3)
```
```
[...]
L2
rho                 rho_v1              rho_v2              rho_e
error     EOC       error     EOC       error     EOC       error     EOC
8.52e-07  -         1.24e-06  -         1.24e-06  -         4.28e-06  -
6.49e-08  3.71      8.38e-08  3.88      8.38e-08  3.88      2.96e-07  3.85      
4.33e-09  3.91      5.39e-09  3.96      5.39e-09  3.96      1.93e-08  3.94

mean      3.81      mean      3.92      mean      3.92      mean      3.90
--------------------------------------------------------------------------------
Linf
rho                 rho_v1              rho_v2              rho_e
error     EOC       error     EOC       error     EOC       error     EOC       
8.36e-06  -         1.03e-05  -         1.03e-05  -         4.50e-05  -
5.58e-07  3.90      6.58e-07  3.97      6.58e-07  3.97      2.92e-06  3.94
3.77e-08  3.89      4.42e-08  3.90      4.42e-08  3.90      1.91e-07  3.93

mean      3.90      mean      3.93      mean      3.93      mean      3.94
--------------------------------------------------------------------------------
```


## Referencing
If you use Trixi in your own research or write a paper using results obtained
with the help of Trixi, please cite the following
[reference](https://arxiv.org/abs/2008.10593):
```bibtex
@online{schlottkelakemper2020purely,
  title={A purely hyperbolic discontinuous {G}alerkin approach for
         self-gravitating gas dynamics},
  author={Schlottke-Lakemper, Michael and Winters, Andrew R and
          Ranocha, Hendrik and Gassner, Gregor J},
  year={2020},
  month={08},
  eprint={2008.10593},
  eprinttype={arXiv},
  eprintclass={math.NA}
}
```

## [Authors](@id authors-index-md)
Trixi was initiated by [Michael
Schlottke-Lakemper](https://www.mi.uni-koeln.de/NumSim/schlottke-lakemper) and
[Gregor Gassner](https://www.mi.uni-koeln.de/NumSim/gregor-gassner) (both
University of Cologne, Germany). Together with [Hendrik Ranocha](https://ranocha.de)
(KAUST, Saudi Arabia) and [Andrew Winters](https://liu.se/en/employee/andwi94)
(Linköping University, Sweden), they are the principal developers of Trixi.
The full list of contributors can be found under [Authors](@ref).


## License and contributing
Trixi is licensed under the MIT license (see [License](@ref)). Since Trixi is
an open-source project, we are very happy to accept contributions from the
community. Please refer to [Contributing](@ref) for more details.
