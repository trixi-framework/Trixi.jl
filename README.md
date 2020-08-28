# Trixi.jl

[![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://trixi-framework.github.io/Trixi.jl/stable)
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://trixi-framework.github.io/Trixi.jl/dev)
[![Build Linux & macOS](https://travis-ci.com/trixi-framework/Trixi.jl.svg?branch=master)](https://travis-ci.com/trixi-framework/Trixi.jl)
[![Build Windows](https://ci.appveyor.com/api/projects/status/uu0xds4hyc1i10n8/branch/master?svg=true)](https://ci.appveyor.com/project/ranocha/trixi-jl/branch/master)
[![Codecov](https://codecov.io/gh/trixi-framework/Trixi.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/trixi-framework/Trixi.jl)
[![Coveralls](https://coveralls.io/repos/github/trixi-framework/Trixi.jl/badge.svg?branch=master)](https://coveralls.io/github/trixi-framework/Trixi.jl?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3996439.svg)](https://doi.org/10.5281/zenodo.3996439)
[![GitHub commits since tagged version](https://img.shields.io/github/commits-since/trixi-framework/Trixi.jl/v0.1.2.svg?style=social&logo=github)](https://github.com/trixi-framework/Trixi.jl)

<p align="center">
  <img width="300px" src="docs/src/assets/logo.png">
</p>

**Trixi.jl** is a numerical simulation framework for hyperbolic conservation
laws written in [Julia](https://julialang.org). A key goal for Trixi is to be
easy to use for new or unexperienced users, including installation and
postprocessing.  Its features include:

* Hierarchical quadtree/octree grid with adaptive mesh refinement
* Native support for 2D and 3D simulations
* High-order accuracy in space in time
* Nodal discontinuous Galerkin spectral element methods
  * Kinetic energy-preserving and entropy-stable split forms
  * Entropy-stable shock capturing
* Explicit low-storage Runge-Kutta time integration
* Square/cubic domains with periodic and Dirichlet boundary conditions
* Multiple governing equations:
  * Compressible Euler equations
  * Magnetohydrodynamics equations
  * Hyperbolic diffusion equations for elliptic problems
  * Scalar advection
* Multi-physics simulations
  * [Self-gravitating gas dynamics](https://github.com/trixi-framework/paper-self-gravitating-gas-dynamics)
* Shared-memory parallelization via multithreading
* Visualization of results with Julia-only tools (2D) or ParaView (2D/3D)


## Installation
If you have not yet installed Julia, please follow the instructions for your
operating system found [here](https://julialang.org/downloads/platform/). Trixi
works with Julia v1.5.

You can then install Trixi, the postprocessing tools, and the respective dependencies by
performing the following steps:

  1. Clone the repository:
     ```bash
     git clone git@github.com:trixi-framework/Trixi.jl.git
     ```
  2. Enter the cloned directory and run the following command to install all
     required dependencies:
     ```bash
     julia utils/install.jl
     ```

Trixi is also a registered Julia package. Hence, you can also install Trixi via
```julia
julia> import Pkg

julia> Pkg.add("Trixi")
```
If you do this and want to modify Trixi, you can run
```julia
julia> Pkg.dev("Trixi") # get a clone of the git repository, usually in ~/.julia/dev/Trixi
```


## Usage
Enter the root directory `Trixi.jl/` and execute
```bash
julia --project=@.
```
This will start an interactive Julia session (REPL) using the project setup
of Trixi.jl. If you have installed Trixi.jl in your default project environment,
you can just start Julia as usual
```bash
julia
```
In the Julia REPL, you need to load the package Trixi
```julia
julia> using Trixi
```
To run a simulation, execute
```julia
Trixi.run("examples/parameters.toml")
```
You can also pass a different parameters file or edit `examples/parameters.toml` to
modify the simulation setup.

Sometimes it can be helpful to run Trixi non-interactively in batch mode, e.g.,
when starting a simulation from another script. This is possible by directly passing
the code that shall be executed to Julia
```bash
julia -e 'using Trixi; Trixi.run("examples/parameters.toml")'
```


## Documentation
Additional documentation is available that contains more information on how to
use Trixi interactively, how to visualize output files etc. It also includes a
section on our preferred development workflow and some tips for using Git. The
latest documentation can be accessed either
[online](https://trixi-framework.github.io/Trixi.jl/dev) or under [`docs/src`](docs/src).


## Referencing
If you use Trixi in your own research or write a paper using results obtained
with the help of Trixi, please cite the following
[paper](https://arxiv.org/abs/2008.10593):
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

In addition, you can also refer to Trixi directly as
```bibtex
@misc{schlottkelakemper2020trixi,
  title={{T}rixi.jl: A flexible tree-based numerical simulation framework
         for {PDE}s written in {J}ulia},
  author={Schlottke-Lakemper, Michael and Gassner, Gregor J and
          Ranocha, Hendrik and Winters, Andrew R},
  year={2020},
  month={08},
  howpublished={\url{https://github.com/trixi-framework/Trixi.jl}},
  doi={10.5281/zenodo.3996439}
}
```


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
