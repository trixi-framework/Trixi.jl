# Trixi.jl

[![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://trixi-framework.github.io/Trixi.jl/stable)
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://trixi-framework.github.io/Trixi.jl/dev)
[![Build Linux & macOS](https://travis-ci.com/trixi-framework/Trixi.jl.svg?branch=master)](https://travis-ci.com/trixi-framework/Trixi.jl)
[![Build Windows](https://ci.appveyor.com/api/projects/status/uu0xds4hyc1i10n8/branch/master?svg=true)](https://ci.appveyor.com/project/ranocha/trixi-jl/branch/master)
[![Coveralls](https://coveralls.io/repos/github/trixi-framework/Trixi.jl/badge.svg?branch=master)](https://coveralls.io/github/trixi-framework/Trixi.jl?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3996439.svg)](https://doi.org/10.5281/zenodo.3996439)
[![GitHub commits since tagged version](https://img.shields.io/github/commits-since/trixi-framework/Trixi.jl/v0.2.5.svg?style=social&logo=github)](https://github.com/trixi-framework/Trixi.jl)
<!-- [![PkgEval](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/T/Trixi.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html) -->
<!-- [![Codecov](https://codecov.io/gh/trixi-framework/Trixi.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/trixi-framework/Trixi.jl) -->

<p align="center">
  <img width="300px" src="docs/src/assets/logo.png">
</p>

**Trixi.jl** is a numerical simulation framework for hyperbolic conservation
laws written in [Julia](https://julialang.org). A key objective for the
framework is to be useful to both scientists and students. Therefore, next to
having an extensible design with a fast implementation, Trixi is
focused on being easy to use for new or inexperienced users, including the
installation and postprocessing procedures. Its features include:

* Hierarchical quadtree/octree grid with adaptive mesh refinement
* Native support for 1D, 2D, and 3D simulations
* High-order accuracy in space in time
* Nodal discontinuous Galerkin spectral element methods
  * Kinetic energy-preserving and entropy-stable split forms
  * Entropy-stable shock capturing
  * Positivity-preserving limiting
* Explicit low-storage Runge-Kutta time integration
* Square/cubic domains with periodic and weakly-enforced boundary conditions
* Multiple governing equations:
  * Compressible Euler equations
  * Magnetohydrodynamics equations
  * Hyperbolic diffusion equations for elliptic problems
  * Scalar advection
* Multi-physics simulations
  * [Self-gravitating gas dynamics](https://github.com/trixi-framework/paper-self-gravitating-gas-dynamics)
* Shared-memory parallelization via multithreading
* Visualization of results with Julia-only tools (2D) or ParaView/VisIt (2D/3D)


## Installation
If you have not yet installed Julia, please [follow the instructions for your
operating system](https://julialang.org/downloads/platform/). Trixi works
with Julia v1.5.

### For users
Trixi and related postprocessing tools are registered Julia packages. Hence, you
can install Trixi, [Trixi2Vtk](https://github.com/trixi-framework/Trixi2Vtk.jl),
and [Trixi2Img](https://github.com/trixi-framework/Trixi2Img.jl) by executing
the following commands in the Julia REPL:
```julia
julia> import Pkg

julia> Pkg.add("Trixi"); Pkg.add("Trixi2Vtk"); Pkg.add("Trixi2Img")
```
Note that you can copy and paste all commands to the REPL *including* the leading
`julia>` prompts - they will automatically be stripped away by Julia.

### For developers
If you plan on editing Trixi itself, you can download Trixi locally and run it from
within the cloned directory:
```bash
git clone git@github.com:trixi-framework/Trixi.jl.git
cd Trixi.jl
julia --project=. -e 'import Pkg; Pkg.instantiate()' # Install Trixi's dependencies
julia -e 'import Pkg; Pkg.add("Trixi2Vtk"); Pkg.add("Trixi2Img")' # Install postprocessing tools
```
If you installed Trixi this way, you always have to start Julia with the `--project`
flag set to your local Trixi clone, e.g.,
```bash
julia --project=.
```
Further details can be found in the [documentation](#documentation).


## Usage
In the Julia REPL, first load the package Trixi
```julia
julia> using Trixi
```
Then start a simulation by executing
```julia
julia> Trixi.run(default_example())
```
To visualize the results, load the package Trixi2Img
```julia
julia> using Trixi2Img
```
and generate a contour plot of the results with
```julia
julia> trixi2img(joinpath("out", "solution_000040.h5"), output_directory="out", grid_lines=true)
```
This will create a file `solution_000040_scalar.png` in the `out/` subdirectory
that can be opened with any image viewer:
<p align="center">
  <img width="300px" src="docs/src/assets/solution_000040_scalar_resized.png">
</p>

The method `Trixi.run(...)` expects a single string argument with the path to a
Trixi parameter file. To quickly see Trixi in action, `default_example()`
returns the path to an example parameter file with a short, two-dimensional
problem setup. A list of all example parameter files packaged with Trixi can be
obtained by running `get_examples()`. Alternatively, you can also browse the
[`examples/`](examples/) subdirectory. If you want to
modify one of the parameter files to set up your own simulation, download it to
your machine, edit the configuration, and pass the file path to `Trixi.run(...)`.

*Note on performance:* Julia uses just-in-time compilation to transform its
source code to native, optimized machine code at the *time of execution* and
caches the compiled methods for further use. That means that the first execution
of a Julia method is typically slow, with subsequent runs being much faster. For
instance, in the example above the first execution of `Trixi.run` takes about 15
seconds, while subsequent runs require less than 50 *milli*seconds.


## Documentation
Additional documentation is available that contains more information on how to
modify/extend Trixi's implementation, how to visualize output files etc. It
also includes a section on our preferred development workflow and some tips for
using Git. The latest documentation can be accessed either
[online](https://trixi-framework.github.io/Trixi.jl/stable) or under [`docs/src`](docs/src).


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
  title={{T}rixi.jl: A tree-based numerical simulation framework
         for hyperbolic {PDE}s written in {J}ulia},
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
