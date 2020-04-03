# Trixi: A tree-based flexible DG/SBP framework written in Julia

<p align="center">
  <img width="300px"
       src="https://gitlab.mi.uni-koeln.de/numsim/code/Trixi.jl/-/raw/master/doc/images/trixi.png">
</p>

*Trixi* is a flexible DG/SBP framework written in the Julia programming
language. It is based on a two-dimensional hierarchical mesh (quadtree) and aims
to be easy to use and extend.

**Table of contents**

*   [Installation](#installation)
*   [Usage](#usage)
*   [Development](#development)
*   [Visualization and postprocessing](#visualization-and-postprocessing)
*   [Style guide](#style-guide)
*   [Authors](#authors)


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


## Development
When a Julia program is executed, the just-in-time compiler has to compile all
functions at their first use, which incurs an overhead each time a program is
run. For proper packages and commands executed in the REPL (= "return-eval-print
loop", which is what the Julia community calls the shell prompt that opens
when running `julia` without any files), however, the previously compiled
functions are cached. Therefore, it can be beneficial to run Trixi from the REPL
during development, as it allows much faster turnaround times.

If you naively run Trixi from the REPL by including `Trixi.jl/src/Trixi.jl`, you will not be
able to change your Trixi source files and then run the changed code without
restarting the REPL, which destroys any potential benefits from caching.
However, restarting Julia can be avoided by using the `Revise` package, which
tracks changed files and re-loads them automatically. Therefore, you first need
to install the `Revise` package using the following command:

```bash
julia -e 'import Pkg; Pkg.add("Revise")'
```

Now you are able to run Trixi from the REPL, change Trixi code between runs,
**and** enjoy the advantages of the compilation cache!


### Automatically starting in interactive mode
To automatically start into an interactive session, run Trixi or one of the
postprocessing tools with the `--interactive` (or short: `-i`) flag. This will
open up a REPL and load everything you need to start using Trixi from the REPL.
When using interactive mode, all command line arguments except
`--interactive`/`-i` are ignored.  You thus have to supply all command line
arguments via the respective `run()` method. The first run will be a little bit
slower (i.e., as when running `bin/trixi` directly), since Julia has to compile
all functions for the first time. Starting with the second run, only those
functions are recompiled for which the source code has changed since the last
invocation.

Please note that `bin/trixi` and the tools in `postprocessing/` all
start the REPL with the `--project` flag set to the respective Trixi project.
That means that if you start in interactive mode and install/remove packages, it
will *only* affect your local Trixi installation and *not* your general Julia
environment. Therefore, to add new packages (such as `Revise`) to your Julia
installation and not just to Trixi, you need to start the REPL manually and
without providing the `--project` flag.

#### Example
Begin by executing:
```bash
bin/trixi -i
```
This will start the Julia REPL with the following output:
```bash
# Note: project directory set to '.'. Changes to packages will only affect current project.
# Execute the first line below once at the beginning of an interactive session.
# Start a Trixi simulation by running the second line.

using Revise; import Trixi
Trixi.run("parameters.toml")
julia>
```
Copy-pasting and then executing the first line yields:
```bash
julia> using Revise; import Trixi
[ Info: Precompiling Trixi [a7f1ee26-1774-49b1-8366-f1abc58fbfcb]

```
You can then proceed by running the second line (you probably want to change
`parameters.toml` to the parameters file you intend to use):
```bash
julia> Trixi.run("parameters.toml")
```


### Manually starting in interactive mode

To manually start in interactive mode (e.g., to supply additional arguments to
the `julia` executable at startup`), execute Julia with the project directory
set to the package directory of the program/tool you want to use:
*   Trixi: `Trixi.jl/`
*   `trixi2vtk`: `Trixi.jl/postprocessing/pgk/Trixi2Vtk`
*   `trixi2img`: `Trixi.jl/postprocessing/pgk/Trixi2Img`

For example, to run Trixi this way, you need to start the REPL with
```bash
julia --project=path/to/Trixi.jl/
```
Then you can just proceed with the usual commands to load and run Trixi as in
the example [above](#example). The `--project` flag is required such that Julia
can properly load Trixi and all her dependencies.


## Visualization and postprocessing
There are two tools provided with Trixi that allow to visualize Trixi's output
files, both of which can be found in `postprocessing/`: `trixi2vtk` and
`trixi2img`. `

### `trixi2vtk` (ParaView-based visualization)
`trixi2vtk` converts Trixi's `.h5` output files to VTK files, which can be read
by [ParaView](https://www.paraview.org) and other visualization tools. It
automatically interpolates solution data from the original quadrature node
locations to equidistant "visualization nodes" at a higher resolution, to make
up for the loss of accuracy from going from a high-order polynomial
representation to a piecewise constant representation in ParaView.

Then, to convert a file, just call `trixi2vtk` with the name of a `.h5` file as argument:
```bash
postprocessing/trixi2vtk out/solution_000000.h5
```
This allows you to generate VTK files for solution, restart and mesh files. By
default, `trixi2vtk` generates `.vtu` (unstructured VTK) files for both cell/element data (e.g.,
cell ids, element ids) and node data (e.g., solution variables). This format
visualizes each cell with the same number of nodes, independent of its size.
Alternatively, you can provide `-f vti` on the command line, which causes
`trixi2vtk` to generate `.vti` (image data VTK) files for the solution files,
while still using `.vtu` files for cell-/element-based data. In `.vti` files,
a uniform resolution is used throughout the entire domain, resulting in
different number of visualization nodes for each element.
This can be advantageous to create publication-quality images, but
increases the file size.

If you want to convert multiple solution/restart files at once, you can just supply
multiple input files on the command line. `trixi2vtk` will then also generate a
`.pvd` file, which allows ParaView to read all `.vtu`/`.vti` files at once and which
uses the `time` attribute in solution/restart files to inform ParaView about the
solution time. To list all command line options,
run `trixi2vtk --help`.

Similarly to Trixi, `trixi2vtk` supports an interactive mode that can be invoked
by running
```bash
postprocessing/trixi2vtk -i
```


### `trixi2img` (Julia-based visualization)
`trixi2img` can be used to directly convert Trixi's output files to image files,
without having to use a third-pary visualization tool such as ParaView. The
downside of this approach is that it generally takes longer to visualize the
data (especially for large files) and that it does not allow to customize the
output without having to directly edit the source code of `trixi2img`.
Currently, PNG and PDF are supported as output formats.

Then, to convert a file, just call `trixi2img` with the name of a `.h5` file as argument:
```bash
postprocessing/trixi2img out/solution_000000.h5
```
Multiple files can be converted at once by specifying more input files on the
command line.

Similarly to Trixi, `trixi2img` supports an interactive mode that can be invoked
by running
```bash
postprocessing/trixi2img -i
```


## Style guide
The following lists a few conventions that have been used so far:

*   Modules, types, structs with `CamelCase`
*   Functions, variables with lowercase `snake_case`  
*   Indentation with 2 spaces (never tabs!), line continuations indented with 4
    spaces
*   Maximum line length (strictly): **100**
*   Prefer `for i in ...` to `for i = ...` for better semantic clarity and
    greater flexibility

Based on that, and personal experience, a formatting tool with a few helpful
options is included in `utils/julia-format.jl`. Note, however, that this tool is
not yet optimal, as it re-indents too greedily.

This is a list of handy style guides that are mostly consistent with each
other and this guide, and which have been used as a basis:

*   https://www.juliaopt.org/JuMP.jl/stable/style/
*   https://github.com/jrevels/YASGuide

## Authors
Trixi was created by Michael Schlottke-Lakemper and Gregor Gassner.
