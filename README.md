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
Enter the root directory `Trixi.jl/` and run
```bash
bin/trixi parameters.toml
```

To change the simulation setup, edit `parameters.toml`. You can also pass a different
parameters file on the command line, e.g., `bin/trixi awesome_parameters.toml`.


## Documentation
Additional documentation is available that contains more information on how to
use Trixi interactively, how to visualize output files etc. It also includes a
section on our preferred development workflow and some tips for using Git. The
documentation can be accessed either
[online](https://numsim.gitlab-pages.sloede.com/code/Trixi.jl/) or under
[`docs/src`](docs/src).


## Authors
Trixi was created by
[Michael Schlottke-Lakemper](https://www.mi.uni-koeln.de/NumSim/schlottke-lakemper) and
[Gregor Gassner](https://www.mi.uni-koeln.de/NumSim/gregor-gassner).
