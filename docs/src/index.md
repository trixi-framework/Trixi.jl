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
     git clone git@gitlab.mi.uni-koeln.de:numsim/code/Trixi.jl.git
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
Trixi.run("parameters.toml")
```
You can also pass a different parameters file or edit `parameters.toml` to
modify the simulation setup.
More information on how to use Trixi interactively can be found in the
[Development](@ref) section.

Sometimes it can be helpful to run Trixi non-interactively in batch mode, e.g., when starting
a simulation from another script. This is possible by directly passing the
parameters file to Trixi on the command line:
```bash
bin/trixi parameters.toml
```

### Example: Running Trixi interactively
```@raw html
  <script id="asciicast-zn79qrdAfCDGWKlQgWHzc0wCB"
          src="https://asciinema.org/a/zn79qrdAfCDGWKlQgWHzc0wCB.js"
          async
          data-cols=90
          ata-rows=20></script>
```


## Authors
Trixi was created by
[Michael Schlottke-Lakemper](https://www.mi.uni-koeln.de/NumSim/schlottke-lakemper) and
[Gregor Gassner](https://www.mi.uni-koeln.de/NumSim/gregor-gassner).
