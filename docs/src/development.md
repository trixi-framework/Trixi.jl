# Development

## Interactive use of Julia
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
However, restarting Julia can be avoided by using the
[Revise.jl](https://github.com/timholy/Revise.jl) package, which tracks changed
files and re-loads them automatically. Therefore, you first need to install
Revise with the following command:

```bash
julia -e 'import Pkg; Pkg.add("Revise")'
```

Now you are able to run Trixi from the REPL, change Trixi code between runs,
**and** enjoy the advantages of the compilation cache! However, before you start using
Revise regularly, please be aware of some of the [Pitfalls when using Revise](@ref).


### Automatically starting Trixi in interactive mode
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
[ Info: Precompiling Trixi [a7f1ee26-1774-49b1-8366-f1abc58fbfcb]
┌ Info: Revise initialized: changes to Trixi source code are tracked.
│ Project directory set to '.'. Adding/removing packages will only affect this project.
│
│ Execute the following line to start a Trixi simulation:
│
└ Trixi.run("parameters.toml")
julia>
```
Proceed by copy-pasting and then executing the last line (you probably want to
change `parameters.toml` to the parameters file you intend to use):
```bash
julia> Trixi.run("parameters.toml")
```


### Manually starting Trixi in interactive mode
To manually start in interactive mode (e.g., to supply additional arguments to
the `julia` executable at startup`), execute Julia with the project directory
set to the package directory of the program/tool you want to use:
  * Trixi: `Trixi.jl/`
  * `trixi2vtk`: `Trixi.jl/postprocessing/pgk/Trixi2Vtk`
  * `trixi2img`: `Trixi.jl/postprocessing/pgk/Trixi2Img`

For example, to run Trixi this way, you need to start the REPL with
```bash
julia --project=path/to/Trixi.jl/
```
Then you can just proceed with the usual commands to load and run Trixi as in
the example [above](#example). The `--project` flag is required such that Julia
can properly load Trixi and all her dependencies.


### Pitfalls when using Revise
While Revise is a great help for developing Julia code, there are a few
situations to watch out for when using Revise. The following list of potential
issues is based on personal experiences of the Trixi developers and probably
incomplete.  Further information on limitations and possible issues with Revise
can be found in the official [documentation](https://timholy.github.io/Revise.jl/stable/).

!!! tip "If in doubt, restart the REPL"
    Oftentimes, it is possible to recover from issues with Revise by fixing the
    offending code. Sometimes, however, this is not possible or you might have
    troubles finding out what exactly caused the problems. Therefore, in these
    cases, or if in doubt, restart the REPL to get a fresh start.

#### Syntax errors are easy to miss
Revise does not stop on syntax errors, e.g., when you accidentally write
`a[i)` instead of `a[i]`.  In this case, Revise reports an error but **continues
to use the old version of your files**! This is especially dangerous for syntax
errors, as they are detected while Revise reloads changed code, which happens in
the beginning of a new execution. Thus, the syntax error message quickly
disappears from the terminal once Trixi starts writing output to the screen and
you might not even have noticed that an error occurred at all.

Therefore, when you are deep in a coding/debugging session and wonder why your
code modifications do not seem to have any effect, scroll up in your terminal to
check if you missed earlier syntax errors, or - if in doubt - restart your REPL.

#### Files are not tracked after changing branches
Sometimes, Revise stops tracking files when changing the Git branch. That is,
modifications to Trixi's source files will not be reloaded by Revise and thus
have no effect of a currently running REPL session. This issue is
particularly annoying for a developer, since it **does not come with any
warning**!  Therefore, it is good practice to always restart the REPL after
changing branches.

#### Changes to type definitions are not allowed
Revise cannot handle changes to type definitions, e.g., when modifying the
fields in a `struct`. In this case, Revise reports an error and refuses to run
your code unless you undo the modifications. Once you undo the changes, Revise
will usually continue to work as expected again. However, if you want to keep
your type modifications, you need to restart the REPL.


## Text editor
When writing code, the choice of text editor can have a significant impact on
productivity and developer satisfaction. While using the default text editor
of the operating system has its own benefits (specifically the lack of an explicit
installation procure), usually it makes sense to switch to a more
programming-friendly tool. In the following, a few of the many options are
listed and discussed:

### Juno
If you are new to programming or do not have a preference for a text editor
yet, [Juno](https://junolab.org) is a good choice for developing Julia code.
It is based on *Atom*, a sophisticated and widely used editor for software
developers, and is enhanced with several Julia-specific features. Furthermore
and especially helpful for novice programmers, it has a MATLAB-like
appearance with easy and interactive access to the current variables, the
help system, and a debugger.

### Vim or Emacs
Vim and Emacs are both very popular editors that work great with Julia. One
of their advantages is that they are text editors without a GUI and as such
are available for almost any operating system. They also are preinstalled on
virtually all Unix-like systems.  However, Vim and Emacs come with their own,
steep learning curve if they have never been used before. Therfore, if in doubt, it
is probably easier to get started with a classic GUI-based text edito (like
Juno). If you decide to use Vim or Emacs, make sure that you install the
corresponding Vim plugin
[julia-vim](https://github.com/JuliaEditorSupport/julia-vim) or Emacs major
mode [julia-emacs](https://github.com/JuliaEditorSupport/julia-emacs).
