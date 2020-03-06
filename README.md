# Trixi: A tree-based flexible DG/SBP framework written in Julia

<p align="center">

  <img width="300px" src="https://gitlab.mi.uni-koeln.de/numsim/code/Trixi.jl/-/raw/master/doc/images/trixi.png" />
</p>

## Installation
Strictly speaking, no installation is necessary to run Trixi. However, the
simulation program and the postprocessing tools rely on a number of Julia
packages, which need to be available on the respective machine. This can most
easily be achieved by performing the following steps:

1.  Clone the repository:
    ```bash
    git clone git@gitlab.mi.uni-koeln.de:numsim/code/Trixi.jl.git
    ```
2.  Enter the cloned directory and run the Julia command-line tool:
    ```bash
    cd Trixi.jl
    julia
    ```
3.  Switch to the package manager by pressing `]`, activate the current
    directory and then instatiate it:
    ```julia
    julia> ]
    (v1.3) pkg> activate .
    Activating environment at `~/path/to/Trixi.jl/Project.toml`

    (Trixi) pkg> instantiate
    ```


## Usage
Enter the root directory `Trixi.jl` and run
```bash
bin/trixi
```

To change the simulation setup, edit `parameters.toml`. You can pass a different
parameter file on the command line using `-p new_parameters.toml`.


## Development
When a Julia program is executed, the just-in-time compiler has to compile all
functions at their first use, which incurs an overhead each time a program is
run. For proper packages and commands executed in the REPL (= "return-eval-print
loop", which is what the Julia community calls the shell prompt that opens
when running `julia` without any files), however, the previously compiled
functions are cached. Therefore, it can be beneficial to run Trixi from the REPL
during development, as it allows much faster turnaround times.

If you naively run Trixi from the REPL by including `Trixi.jl`, you will not be
able to change your Trixi source files and then run the changed code without
restarting the REPL, which destroys any potential benefits from caching.
However, restarting Julia can be avoided by using the `Revise.jl` package, which
tracks changed files and re-loads them automatically. Therefore, you first need
to install the `Revise.jl` package using the package installation by the
following procedure:

1.  Start `julia`.
2.  Switch to the package manager by pressing `]`
3.  Execute `add Revise`

Now you are able to run Trixi from the REPL, change Trixi code between runs,
**and** enjoy the advantages of the compilation cache:

1.  Go to the Trixi root directory and start the Julia REPL by running `julia`.
2.  From the REPL, load the `Revise` package (this step _has_ to come first, as
    otherwise `Revise` will not be able to track changes to the source code of
    Trixi):
    ```julia
       julia> using Revise
    ```
3.  Add the current directory to the load path for modules:
    ```julia
       julia> push!(LOAD_PATH, ".")
       4-element Array{String,1}:
        "@"      
        "@v#.#"  
        "@stdlib"
        "."      
    ```
4.  Import Trixi, ignoring the warnings shown below:
    ```julia
    julia> import Trixi
    [ Info: Precompiling Trixi [a7f1ee26-1774-49b1-8366-f1abc58fbfcb]
    ┌ Warning: Package Trixi does not have Pkg in its dependencies:
    │ - If you have Trixi checked out for development and have
    │   added Pkg as a dependency but haven't updated your primary
    │   environment's manifest file, try `Pkg.resolve()`.
    │ - Otherwise you may need to report an issue with Trixi
    └ Loading Pkg into Trixi from project dependency, future warnings for Trixi are suppressed.
    ```
5.  Run Trixi by calling its `main()` function with the parameters file as a
    keyword argument:
    ```julia
    julia> Trixi.run(parameters_file="parameters.toml")
    ```

The first run will be a little bit slower (i.e., as when running `bin/trixi`
directly), as Julia has to compile all functions for the first time. Starting at
the second run, only those functions are recompiled for which the source code
has changed since the last invocation.

If you wish to use `Revise` to run a script that is not a package, e.g., the
plotting tools in `postprocessing/`, you cannot use `import` but need to include
the script directly using `includet(...)`. For example, for the `plot2d.jl`
script, you would perform the following steps:
```julia
julia> using Revise

julia> includet("postprocessing/plotfast.jl")

julia> ┌ Warning: /home/mschlott/.julia/packages/Plots/12uaJ/src/Plots.jl/ is not an existing directory, Revise is not watching
└ @ Revise /home/mschlott/.julia/packages/Revise/SZ4ae/src/Revise.jl:489

julia> TrixiPlot.main()
```
Once again, you can usually safely ignore the warning.


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
