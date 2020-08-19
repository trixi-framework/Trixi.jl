# Trixi.jl

[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://trixi-framework.github.io/Trixi.jl/dev)
[![Build Linux & macOS](https://travis-ci.com/trixi-framework/Trixi.jl.svg?branch=master)](https://travis-ci.com/trixi-framework/Trixi.jl)
[![Build Windows](https://ci.appveyor.com/api/projects/status/uu0xds4hyc1i10n8/branch/master?svg=true)](https://ci.appveyor.com/project/ranocha/trixi-jl/branch/master)
[![Codecov](https://codecov.io/gh/trixi-framework/Trixi.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/trixi-framework/Trixi.jl)
[![Coveralls](https://coveralls.io/repos/github/trixi-framework/Trixi.jl/badge.svg?branch=master)](https://coveralls.io/github/trixi-framework/Trixi.jl?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

<!-- [![GitHub commits since tagged version](https://img.shields.io/github/commits-since/trixi-framework/Trixi.jl/v0.1.0.svg?style=social&logo=github)](https://github.com/trixi-framework/Trixi.jl) -->
<!-- [![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://trixi-framework.github.io/Trixi.jl/stable) -->
<!-- [![DOI](https://zenodo.org/badge/DOI/TODO.svg)](https://doi.org/TODO) -->


**A flexible, tree-based numerical simulation framework for PDEs written in Julia.**

<p align="center">
  <img width="300px" src="docs/src/assets/logo.png">
</p>

**Trixi.jl** is a flexible numerical simulation framework for partial
differential equations. It is based on a two-dimensional hierarchical mesh
(quadtree) and supports several governing equations such as compressible Euler
equations, magnetohydrodynamics equations, or hyperbolic diffusion equations.
Trixi is written in [Julia](https://julialang.org) and aims to be easy to use and
extend also for new or inexperienced users.


## Installation
If you have not yet installed Julia, please follow the instructions for your
operating system found [here](https://julialang.org/downloads/platform/). Trixi
works with Julia v1.5.

You can then install Trixi, the postprocessing tools, and the respective dependencies by
performing the following steps:

  1. Clone the repository:
     ```
     git clone git@github.com:trixi-framework/Trixi.jl.git
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
Trixi.run("examples/parameters.toml")
```
You can also pass a different parameters file or edit `examples/parameters.toml` to
modify the simulation setup.

Sometimes it can be helpful to run Trixi non-interactively in batch mode, e.g., when starting
a simulation from another script. This is possible by directly passing the
parameters file to Trixi on the command line:
```bash
bin/trixi examples/parameters.toml
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
[Gregor Gassner](https://www.mi.uni-koeln.de/NumSim/gregor-gassner) (both
University of Cologne, Germany). Together with [Hendrik Ranocha](https://ranocha.de)
(KAUST, Saudi Arabia) and [Andrew Winters](https://liu.se/en/employee/andwi94)
(Linköping University, Sweden), they are the principal developers of Trixi.
The full list of contributors can be found in [AUTHORS.md](AUTHORS.md).


## License and contributing
Trixi is licensed under the MIT license (see [LICENSE.md](LICENSE.md)). Since Trixi is
an open-source project, we are very happy to accept contributions from the
community. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
