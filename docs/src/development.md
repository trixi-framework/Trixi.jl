# Development

## Interactive use of Julia
When a Julia program is executed, Julia first loads and parses all code. Then,
the just-in-time compiler has to compile all functions at their first use, which
incurs an overhead each time a program is run. For proper packages and commands
executed in the REPL (= "return-eval-print loop", which is what the Julia
community calls the interactive command-line prompt that opens when executing
`julia` without any files as arguments), however, the previously compiled functions are
cached. Therefore, Trixi should generally always be used interactively from the
REPL without closing Julia during development, as it allows much faster turnaround times.

If you naively run Trixi from the REPL, you will not be able to change your
Trixi source files and then run the changed code without restarting the REPL,
which destroys any potential benefits from caching. However, restarting
Julia can be avoided by using the [Revise.jl](https://github.com/timholy/Revise.jl)
package, which tracks changed files and re-loads them automatically. Therefore,
it is *highly recommended* to first install Revise with the following command in Julia:
To enter the packe REPL mode, press `]` in the standard Julia REPL mode. Then, execute
```julia
(@v1.5) pkg> add Revise
```
Now you are able to run Trixi from the REPL, change Trixi code between runs,
**and** enjoy the advantages of the compilation cache! Before you start using
Revise regularly, please be aware of some of the [Pitfalls when using Revise](@ref).

Another recommended package for working from the REPL is
[OhMyREPL.jl](https://github.com/KristofferC/OhMyREPL.jl). It can be installed
by running
```julia
(@v1.5) pkg> add OhMyREPL
```
and adds syntax highlighting, bracket highlighting, and other helpful
improvements for using Julia interactively. To automatically use OhMyREPL when
starting the REPL, follow the instructions given in the official
[documentation](https://kristofferc.github.io/OhMyREPL.jl/latest/).

### Running Trixi interactively in the global environment
If you've installed Trixi and Revise in your default environment,
begin by executing:
```bash
julia
```
This will start the Julia REPL. Then, run
```julia
julia> using Revise; using Trixi
```
You can run a simulation by executing
```julia
julia> Trixi.run("examples/2d/parameters.toml")
```
Together, all of these commands can take some time, roughly half a minute on a
modern workstation. Most of the time is spent on compilation of Julia code etc.
If you execute the last command again in the same REPL, it will finish within a
few milliseconds (maybe ~45 on a modern workstation).  This demonstrates the
second reason for using the REPL: the compilation cache.  That is, those parts
of the code that do not change between two Trixi runs do not need to be
recompiled and thus execute much faster after the first run.


### Manually starting Trixi in the local environment
If you followed the [installation instructions for developers](@ref for-developers), execute
Julia with the project directory set to the package directory of the
program/tool you want to use.
For example, to run Trixi this way, you need to start the REPL with
```bash
julia --project=path/to/Trixi.jl/
```
and execute
```julia
julia> using Revise; using Trixi
```
to load Revise and Trixi. You can then proceed with the usual commands and run Trixi as in
the example [above](#Running-Trixi-interactively-in-the-global-environment-1).
The `--project` flag is required such that Julia can properly load Trixi and all dependencies
if Trixi is not installed in the global environment. The same procedure also
applies should you opt to install the postprocessing tools
[Trixi2Vtk](https://github.com/trixi-framework/Trixi2Vtk.jl) and
[Trixi2Img](https://github.com/trixi-framework/Trixi2Img.jl) manually such that
you can modify their implemenations.


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


## Text editors
When writing code, the choice of text editor can have a significant impact on
productivity and developer satisfaction. While using the default text editor
of the operating system has its own benefits (specifically the lack of an explicit
installation procure), usually it makes sense to switch to a more
programming-friendly tool. In the following, a few of the many options are
listed and discussed:

### VS Code
[Visual Studio Code](https://code.visualstudio.com/) is a modern open source
editor with [good support for Julia](https://github.com/julia-vscode/julia-vscode).
While [Juno](#Juno) had some better support in the past, the developers of Juno
and the Julia VS Code plugin are joining forces and concentrating on VS Code
since support of Atom has been suspended. Basically, all comments on [Juno](#Juno)
below also apply to VS Code.

### Juno
If you are new to programming or do not have a preference for a text editor
yet, [Juno](https://junolab.org) is a good choice for developing Julia code.
It is based on *Atom*, a sophisticated and widely used editor for software
developers, and is enhanced with several Julia-specific features. Furthermore
and especially helpful for novice programmers, it has a MATLAB-like
appearance with easy and interactive access to the current variables, the
help system, and a debugger.

When using Juno's REPL to run Trixi, you cannot execute the `bin/trixi` script
to start Trixi interactively. Instead, you can include the file `utils/juno.jl`,
which will set the project path, load Revise (if installed), and import Trixi:
```bash
julia> include("utils/juno.jl")
```
Afterwards, you can start Trixi in the usual way by calling the `Trixi.run` method.

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
