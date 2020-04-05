# Development
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


## Automatically starting in interactive mode
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

### Example
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


## Manually starting in interactive mode

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

