# Trixi: A tree-based flexible DG/SBP framework written in Julia

<p align="center">
  <img width="300px" src="docs/src/assets/logo.png">
</p>

*Trixi* is a flexible DG/SBP framework written in the Julia programming
language. It is based on a two-dimensional hierarchical mesh (quadtree) and aims
to be easy to use and extend.


## Installation
If you have not yet installed Julia, please follow the instructions for your
operating system found [here](https://julialang.org/downloads/platform/). Trixi
works with Julia v1.3 or higher.

You can then install Trixi, the postprocessing tools, and the respective dependencies by
performing the following steps:

  1. Clone the repository:
     ```
     git clone git@gitlab.mi.uni-koeln.de:numsim/code/Trixi.jl.git
     ```
  2. Enter the cloned directory and run the following command to install all
     required dependencies:
     ```
     julia utils/install.jl
     ```


## Usage
Enter the root directory `Trixi.jl/` and execute
```bash
bin/trixi
```
This will start an interactive Julia session with the Trixi module already
loaded. To run a simulation, execute
```julia
Trixi.run("parameters.toml")
```
You can also pass a different parameters file or edit `parameters.toml` to
modify the simulation setup. 

Sometimes it can be helpful to run Trixi non-interactively in batch mode, e.g., when starting
a simulation from another script. This is possible by directly passing the
parameters file to Trixi on the command line:
```bash
bin/trixi parameters.toml
```


## Documentation
Additional documentation is available that contains more information on how to
use Trixi interactively, how to visualize output files etc. It also includes a
section on our preferred development workflow and some tips for using Git. The
documentation can be accessed either
[online](https://numsim.gitlab-pages.sloede.com/code/Trixi.jl/) (restricted
to authorized users) or under [`docs/src`](docs/src).


## Authors
Trixi was initiated by [Michael
Schlottke-Lakemper](https://www.mi.uni-koeln.de/NumSim/schlottke-lakemper) and
[Gregor Gassner](https://www.mi.uni-koeln.de/NumSim/gregor-gassner), both from
the University of Cologne, Germany. They are also the principal developers of Trixi.
The full list of contributors can be found in [AUTHORS.md](AUTHORS.md).


## License and contributing
Trixi is licensed under the MIT license (see [LICENSE](LICENSE)). Since Trixi is
an open-source project, we are very happy to accept contributions from the
community. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
