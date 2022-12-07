# Trixi.jl

[![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://trixi-framework.github.io/Trixi.jl/stable)
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://trixi-framework.github.io/Trixi.jl/dev)
[![Slack](https://img.shields.io/badge/chat-slack-e01e5a)](https://join.slack.com/t/trixi-framework/shared_invite/zt-sgkc6ppw-6OXJqZAD5SPjBYqLd8MU~g)
[![Youtube](https://img.shields.io/youtube/channel/views/UCpd92vU2HjjTPup-AIN0pkg?style=social)](https://www.youtube.com/@trixi-framework)
[![Build Status](https://github.com/trixi-framework/Trixi.jl/workflows/CI/badge.svg)](https://github.com/trixi-framework/Trixi.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/trixi-framework/Trixi.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/trixi-framework/Trixi.jl)
[![Coveralls](https://coveralls.io/repos/github/trixi-framework/Trixi.jl/badge.svg?branch=main)](https://coveralls.io/github/trixi-framework/Trixi.jl?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3996439.svg)](https://doi.org/10.5281/zenodo.3996439)

[**Trixi.jl**](https://github.com/trixi-framework/Trixi.jl)
is a numerical simulation framework for hyperbolic conservation
laws written in [Julia](https://julialang.org). A key objective for the
framework is to be useful to both scientists and students. Therefore, next to
having an extensible design with a fast implementation, Trixi is
focused on being easy to use for new or inexperienced users, including the
installation and postprocessing procedures. Its features include:

* 1D, 2D, and 3D simulations on [line/quad/hex/simplex meshes](https://trixi-framework.github.io/Trixi.jl/stable/overview/#Semidiscretizations)
  * Cartesian and curvilinear meshes
  * Conforming and non-conforming meshes
  * Structured and unstructured meshes
  * Hierarchical quadtree/octree grid with adaptive mesh refinement
  * Forests of quadtrees/octrees with [p4est](https://github.com/cburstedde/p4est) via [P4est.jl](https://github.com/trixi-framework/P4est.jl)
* High-order accuracy in space and time
* Discontinuous Galerkin methods
  * Kinetic energy-preserving and entropy-stable methods based on flux differencing
  * Entropy-stable shock capturing
  * Positivity-preserving limiting
  * Finite difference summation by parts (SBP) methods
* Compatible with the [SciML ecosystem for ordinary differential equations](https://diffeq.sciml.ai/latest/)
  * [Explicit low-storage Runge-Kutta time integration](https://diffeq.sciml.ai/latest/solvers/ode_solve/#Low-Storage-Methods)
  * [Strong stability preserving methods](https://diffeq.sciml.ai/latest/solvers/ode_solve/#Explicit-Strong-Stability-Preserving-Runge-Kutta-Methods-for-Hyperbolic-PDEs-(Conservation-Laws))
  * CFL-based and error-based time step control
* Native support for differentiable programming
  * Forward mode automatic differentiation via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
* Periodic and weakly-enforced boundary conditions
* Multiple governing equations:
  * Compressible Euler equations
  * Magnetohydrodynamics (MHD) equations
  * Multi-component compressible Euler and MHD equations
  * Linearized Euler and acoustic perturbation equations
  * Hyperbolic diffusion equations for elliptic problems
  * Lattice-Boltzmann equations (D2Q9 and D3Q27 schemes)
  * Shallow water equations
  * Scalar advection
* Multi-physics simulations
  * [Self-gravitating gas dynamics](https://github.com/trixi-framework/paper-self-gravitating-gas-dynamics)
* Shared-memory parallelization via multithreading
* Visualization and postprocessing of the results
  * In-situ and a posteriori visualization with [Plots.jl](https://github.com/JuliaPlots/Plots.jl)
  * Interactive visualization with [Makie.jl](https://makie.juliaplots.org/)
  * Postprocessing with ParaView/VisIt via [Trixi2Vtk](https://github.com/trixi-framework/Trixi2Vtk.jl)


## Installation
If you have not yet installed Julia, please [follow the instructions for your
operating system](https://julialang.org/downloads/platform/). Trixi works
with Julia v1.7.

### For users
Trixi and its related tools are registered Julia packages. Hence, you
can install Trixi, the visualization tool
[Trixi2Vtk](https://github.com/trixi-framework/Trixi2Vtk.jl),
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl), and
[Plots.jl](https://github.com/JuliaPlots/Plots.jl)
by executing the following commands in the Julia REPL:
```julia
julia> import Pkg

julia> Pkg.add(["Trixi", "Trixi2Vtk", "OrdinaryDiffEq", "Plots"])
```
You can copy and paste all commands to the REPL *including* the leading
`julia>` prompts - they will automatically be stripped away by Julia.
The package [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
provides time integration schemes used by Trixi, while
[Plots.jl](https://github.com/JuliaPlots/Plots.jl) can be used to directly
visualize Trixi's results from the REPL.

*Note on package versions:* If some of the examples for how to use Trixi do not
work, verify that you are using a recent Trixi release by comparing the
installed Trixi version from
```julia
julia> import Pkg; Pkg.update("Trixi"); Pkg.status("Trixi")
```
to the [latest release](https://github.com/trixi-framework/Trixi.jl/releases/latest).
If the installed version does not match the current release, please check the
[Troubleshooting](@ref old-release) section.

The commands above can also be used to update Trixi. A brief list of notable
changes to Trixi is available in [`NEWS.md`](https://github.com/trixi-framework/Trixi.jl/NEWS.md).

### [For developers](@id for-developers)
If you plan on editing Trixi itself, you have two options: installing it as a
`dev` package or cloning it to a local folder.

1. **Install Trixi as a `dev` package**: You can install Trixi as a `dev` package by running
   ```julia
   julia> import Pkg

   julia> Pkg.develop("Trixi")
   ```
   This will download and install Trixi to a designated package development directory
   (usually `.julia/dev`, but you can determine the path on your system with
   `Pkg.devdir()`). As opposed to using the `Pkg.add` method as described above,
   packages installed via `Pkg.develop` may have their files edited locally.

2. **Install Trixi in a local folder**: Alternatively, you can download Trixi
   locally and use it from within the cloned directory:
   ```bash
   git clone git@github.com:trixi-framework/Trixi.jl.git
   cd Trixi.jl
   julia --project=@. -e 'import Pkg; Pkg.instantiate()' # Install Trixi's dependencies
   ```
   The last line can also be used to *update* the dependencies if they have changed
   since you first installed Trixi.

   If you installed Trixi this way, you always have to start Julia with the `--project`
   flag set to your local Trixi clone, e.g.,
   ```bash
   julia --project=@.
   ```

Either way, since the postprocessing tool Trixi2Vtk typically does not need to be modified,
it is recommended to install it as a normal package by executing
```julia
julia> import Pkg

julia> Pkg.add("Trixi2Vtk")
```
in the Julia REPL. Likewise, you can install
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) and
[Plots.jl](https://github.com/JuliaPlots/Plots.jl)
as ordinary packages with the following REPL commands:
```julia
julia> import Pkg

julia> Pkg.add("OrdinaryDiffEq"); Pkg.add("Plots")
```



### Example: Installing Trixi as a package
```@raw html
  <script id="asciicast-373869"
          src="https://asciinema.org/a/373869.js"
          async
          data-cols=100
          data-rows=20
          data-speed=3></script>
```
Please note that the playback speed is set to 3x, thus the entire installation
procedure lasts around 45 seconds in real time (depending on the performance of
your computer and on how many dependencies had already been installed before).


## Usage
In the Julia REPL, first load the package Trixi
```julia
julia> using Trixi
```
Then start a simulation by executing
```julia
julia> trixi_include(default_example())
```
To visualize the results, load the package Plots
```julia
julia> using Plots
```
and generate a heatmap plot of the results with
```julia
julia> plot(sol) # No trailing semicolon, otherwise no plot is shown
```
This will open a new window with a 2D visualization of the final solution:

![image](https://user-images.githubusercontent.com/26361975/177492363-74cee347-7abe-4522-8b2d-0dfadc317f7e.png)

The method `trixi_include(...)` expects a single string argument with the path to a
Trixi elixir, i.e., a text file containing Julia code necessary to set up and run a
simulation. To quickly see Trixi in action, `default_example()`
returns the path to an example elixir with a short, two-dimensional
problem setup. A list of all example elixirs packaged with Trixi can be
obtained by running `get_examples()`. Alternatively, you can also browse the
[`examples/`](https://github.com/trixi-framework/Trixi.jl/examples/) subdirectory.
If you want to modify one of the elixirs to set up your own simulation,
download it to your machine, edit the configuration, and pass the file path to
`trixi_include(...)`.

### Example: Running a simulation with Trixi
```@raw html
  <script id="asciicast-372987"
          src="https://asciinema.org/a/372987.js"
          async
          data-cols=100
          data-rows=48></script>
```
If this produces weird symbols or question marks in the terminal on your system,
you are probably using Mac OS with problematic fonts. In that case, please check
the [Troubleshooting](@ref font-issues) section.

*Note on performance:* Julia uses just-in-time compilation to transform its
source code to native, optimized machine code at the *time of execution* and
caches the compiled methods for further use. That means that the first execution
of a Julia method is typically slow, with subsequent runs being much faster. For
instance, in the example above the first execution of `trixi_include` takes about
20 seconds, while subsequent runs require less than 60 *milli*seconds.


### Performing a convergence analysis
To automatically determine the experimental order of convergence (EOC) for a
given setup, execute
```julia
julia> convergence_test(default_example(), 4)
```
This will run a convergence test with the elixir `default_example()`,
using four iterations with different initial refinement levels. The initial
iteration will use the elixir unchanged, while for each subsequent
iteration the `initial_refinement_level` parameter is incremented by one.
Finally, the measured ``l^2`` and ``l^\infty`` errors and the determined EOCs
will be displayed like this:
```
[...]
l2
scalar
error     EOC
9.14e-06  -
5.69e-07  4.01
3.55e-08  4.00
2.22e-09  4.00

mean      4.00
--------------------------------------------------------------------------------
linf
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
julia> convergence_test(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_euler_source_terms.jl"), 3)
```
```
[...]
l2
rho                 rho_v1              rho_v2              rho_e
error     EOC       error     EOC       error     EOC       error     EOC
9.32e-07  -         1.42e-06  -         1.42e-06  -         4.82e-06  -
7.03e-08  3.73      9.53e-08  3.90      9.53e-08  3.90      3.30e-07  3.87
4.65e-09  3.92      6.09e-09  3.97      6.09e-09  3.97      2.12e-08  3.96

mean      3.82      mean      3.93      mean      3.93      mean      3.91
--------------------------------------------------------------------------------
linf
rho                 rho_v1              rho_v2              rho_e
error     EOC       error     EOC       error     EOC       error     EOC
9.58e-06  -         1.17e-05  -         1.17e-05  -         4.89e-05  -
6.23e-07  3.94      7.48e-07  3.97      7.48e-07  3.97      3.22e-06  3.92
4.05e-08  3.94      4.97e-08  3.91      4.97e-08  3.91      2.10e-07  3.94

mean      3.94      mean      3.94      mean      3.94      mean      3.93
--------------------------------------------------------------------------------
```

### Showcase of advanced features
The presentation [From Mesh Generation to Adaptive Simulation: A Journey in Julia](https://youtu.be/_N4ozHr-t9E),
originally given as part of JuliaCon 2022, outlines how to use Trixi for an adaptive simulation
of the compressible Euler equations in two spatial dimensions on a complex domain. More details
as well as code to run the simulation presented can be found at the
[reproducibility repository](https://github.com/trixi-framework/talk-2022-juliacon_toolchain)
for the presentation.

## Referencing
If you use Trixi in your own research or write a paper using results obtained
with the help of Trixi, please cite the following articles:
```bibtex
@article{ranocha2022adaptive,
  title={Adaptive numerical simulations with {T}rixi.jl:
         {A} case study of {J}ulia for scientific computing},
  author={Ranocha, Hendrik and Schlottke-Lakemper, Michael and Winters, Andrew Ross
          and Faulhaber, Erik and Chan, Jesse and Gassner, Gregor},
  journal={Proceedings of the JuliaCon Conferences},
  volume={1},
  number={1},
  pages={77},
  year={2022},
  doi={10.21105/jcon.00077},
  eprint={2108.06476},
  eprinttype={arXiv},
  eprintclass={cs.MS}
}

@article{schlottkelakemper2021purely,
  title={A purely hyperbolic discontinuous {G}alerkin approach for
         self-gravitating gas dynamics},
  author={Schlottke-Lakemper, Michael and Winters, Andrew R and
          Ranocha, Hendrik and Gassner, Gregor J},
  journal={Journal of Computational Physics},
  pages={110467},
  year={2021},
  month={06},
  volume={442},
  publisher={Elsevier},
  doi={10.1016/j.jcp.2021.110467},
  eprint={2008.10593},
  eprinttype={arXiv},
  eprintclass={math.NA}
}
```

In addition, you can also refer to Trixi directly as
```bibtex
@misc{schlottkelakemper2020trixi,
  title={{T}rixi.jl: {A}daptive high-order numerical simulations
         of hyperbolic {PDE}s in {J}ulia},
  author={Schlottke-Lakemper, Michael and Gassner, Gregor J and
          Ranocha, Hendrik and Winters, Andrew R and Chan, Jesse},
  year={2021},
  month={09},
  howpublished={\url{https://github.com/trixi-framework/Trixi.jl}},
  doi={10.5281/zenodo.3996439}
}
```

## [Authors](@id authors-index-md)
Trixi was initiated by [Michael
Schlottke-Lakemper](https://lakemper.eu)
(RWTH Aachen University, Germany) and
[Gregor Gassner](https://www.mi.uni-koeln.de/NumSim/gregor-gassner)
(University of Cologne, Germany). Together with [Hendrik Ranocha](https://ranocha.de)
(University of Hamburg, Germany) and [Andrew Winters](https://liu.se/en/employee/andwi94)
(Linköping University, Sweden), and [Jesse Chan](https://jlchan.github.io) (Rice University, US),
they are the principal developers of Trixi.
The full list of contributors can be found under [Authors](@ref).


## License and contributing
Trixi is licensed under the MIT license (see [License](@ref)). Since Trixi is
an open-source project, we are very happy to accept contributions from the
community. Please refer to [Contributing](@ref) for more details.
To get in touch with the developers,
[join us on Slack](https://join.slack.com/t/trixi-framework/shared_invite/zt-sgkc6ppw-6OXJqZAD5SPjBYqLd8MU~g)
or [create an issue](https://github.com/trixi-framework/Trixi.jl/issues/new).


## Acknowledgments
This project has benefited from funding by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)
under Germany's Excellence Strategy EXC 2044-390685587, Mathematics Münster:
Dynamics-Geometry-Structure.

This project has benefited from funding from the European Research Council through the
ERC Starting Grant "An Exascale aware and Un-crashable Space-Time-Adaptive
Discontinuous Spectral Element Solver for Non-Linear Conservation Laws" (Extreme),
ERC grant agreement no. 714487.

This project has benefited from funding from Vetenskapsrådet (VR, Swedish Research Council), Sweden grant
agreement 2020-03642 VR.

This project has benefited from funding from the United States National Science Foundation under awards
DMS-1719818 and DMS-1943186.
