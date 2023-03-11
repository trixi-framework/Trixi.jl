# Development

## [Interactive use of Julia](@id interactive-use-of-julia)
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
To enter the package REPL mode, press `]` in the standard Julia REPL mode. Then, execute
```julia-repl
(@v1.7) pkg> add Revise
```
Now you are able to run Trixi from the REPL, change Trixi code between runs,
**and** enjoy the advantages of the compilation cache! Before you start using
Revise regularly, please be aware of some of the [Pitfalls when using Revise](@ref).

Another recommended package for working from the REPL is
[OhMyREPL.jl](https://github.com/KristofferC/OhMyREPL.jl). It can be installed
by running
```julia-repl
(@v1.7) pkg> add OhMyREPL
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
```julia-repl
julia> using Revise; using Trixi
```
You can run a simulation by executing
```julia-repl
julia> trixi_include(default_example())
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
```julia-repl
julia> using Revise; using Trixi
```
to load Revise and Trixi. You can then proceed with the usual commands and run Trixi as in
the example [above](#Running-Trixi-interactively-in-the-global-environment-1).
The `--project` flag is required such that Julia can properly load Trixi and all dependencies
if Trixi is not installed in the global environment. The same procedure also
applies should you opt to install the postprocessing tool
[Trixi2Vtk](https://github.com/trixi-framework/Trixi2Vtk.jl)
manually such that you can modify their implementations.


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


## Using the Julia REPL effectively
The [Julia manual](https://docs.julialang.org/en/v1/manual/getting-started/)
is an excellent resource to learn Julia. Here, we list some helpful commands
than can increase your productivity in the Julia REPL.

- Use the [REPL help mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Help-mode)
  entered by typing `?`.
  ```julia-repl
  julia> using Trixi

  help?> trixi_include
  search: trixi_include

    trixi_include([mod::Module=Main,] elixir::AbstractString; kwargs...)

    include the file elixir and evaluate its content in the global scope of module mod. You can override specific
    assignments in elixir by supplying keyword arguments. It's basic purpose is to make it easier to modify some
    parameters while running Trixi from the REPL. Additionally, this is used in tests to reduce the computational
    burden for CI while still providing examples with sensible default values for users.

    Examples
    ≡≡≡≡≡≡≡≡≡≡

    julia> trixi_include(@__MODULE__, default_example(), tspan=(0.0, 0.1))
    [...]

    julia> sol.t[end]
    0.1
  ```
- You can copy and paste REPL history including `julia>` prompts into the REPL.
- Use tab completion in the REPL, both for names of functions/types/variables and
  for function arguments.
  ```julia-repl
  julia> flux_ranocha( # and TAB
  flux_ranocha(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations1D) in Trixi at ~/.julia/dev/Trixi/src/equations/compressible_euler_1d.jl:390
  flux_ranocha(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D) in Trixi at ~/.julia/dev/Trixi/src/equations/compressible_euler_2d.jl:839
  [...]
  ```
- Use `methodswith` to discover methods associated to a given type etc.
  ```julia-repl
  julia> methodswith(CompressibleEulerEquations2D)
  [1] initial_condition_convergence_test(x, t, equations::CompressibleEulerEquations2D) in Trixi at ~/.julia/dev/Trixi/src/equations/compressible_euler_2d.jl:51
  [...]
  ```
- Use `@which` (or `@edit`) for method calls.
  ```julia-repl
  julia> @which trixi_include(default_example())
  trixi_include(elixir::AbstractString; kwargs...) in Trixi at ~/.julia/dev/Trixi/src/auxiliary/special_elixirs.jl:36
  ```
- Use `apropos` to search through the documentation and docstrings.
  ```julia-repl
  julia> apropos("MHD")
  Trixi.IdealGlmMhdEquations3D
  Trixi.IdealGlmMhdMulticomponentEquations2D
  Trixi.calc_fast_wavespeed_roe
  Trixi.IdealGlmMhdEquations1D
  Trixi.initial_condition_constant
  Trixi.flux_nonconservative_powell
  Trixi.GlmSpeedCallback
  Trixi.flux_derigs_etal
  Trixi.flux_hindenlang_gassner
  Trixi.initial_condition_convergence_test
  Trixi.min_max_speed_naive
  Trixi.IdealGlmMhdEquations2D
  Trixi.IdealGlmMhdMulticomponentEquations1D
  [...]
  ```


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

### Vim or Emacs
Vim and Emacs are both very popular editors that work great with Julia. One
of their advantages is that they are text editors without a GUI and as such
are available for almost any operating system. They also are preinstalled on
virtually all Unix-like systems.  However, Vim and Emacs come with their own,
steep learning curve if they have never been used before. Therefore, if in doubt, it
is probably easier to get started with a classic GUI-based text editor (like
Juno). If you decide to use Vim or Emacs, make sure that you install the
corresponding Vim plugin
[julia-vim](https://github.com/JuliaEditorSupport/julia-vim) or Emacs major
mode [julia-emacs](https://github.com/JuliaEditorSupport/julia-emacs).


## Debugging
Julia offers several options for debugging. A classical debugger is available with the
[Debugger.jl](https://github.com/JuliaDebug/Debugger.jl) package or in the
[Julia extension for VS Code](https://www.julia-vscode.org/docs/stable/userguide/debugging/).
However, it can be quite slow and, at the time of writing (January 2023), currently does not work
properly with Trixi. The [Infiltrator.jl](https://github.com/JuliaDebug/Infiltrator.jl) package on
the other hand does not offer all features of a full debugger, but is a fast and simple tool that
allows users to set breakpoints to open a local REPL session and access the call stack and variables.

### Infiltrator
The Infiltrator package provides fast, interactive breakpoints using the ```@infiltrate``` command,
which drops the user into a local REPL session. From there, it is possible to access local variables,
see the call stack, and execute statements.

The package can be installed in the Julia REPL by executing
```julia-repl
(@v1.8) pkg> add Infiltrator
```

To load the package in the Julia REPL execute
```julia-repl
julia> using Infiltrator
```

Breakpoints can be set by adding a line with the ```@infiltrate``` macro at the respective position
in the code. Use [Revise](@ref interactive-use-of-julia) if you want to set and delete breakpoints
in your package without having to restart Julia.

!!! note
    When running Julia inside a package environment, the ```@infiltrate``` macro only works if `Infiltrator`
    has been added to the dependencies. Another work around when using Revise is to first load the
    package and then add breakpoints with `Main.@infiltrate` to the code. If this is not
    desired, the functional form
    ```julia
    if isdefined(Main, :Infiltrator)
      Main.Infiltrator.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
    end
    ```
    can be used to set breakpoints when working with Trixi or other packages.

Triggering the breakpoint starts a REPL session where it is possible to interact with the current
local scope. Possible commands are:
- ```@locals```: Print the local variables.
- ```@exfiltrate```: Save the local variables to a global storage, which can be accessed with
  the ```safehouse``` variable outside the Infiltrator session.
- ```@trace```: Print the current stack trace.
- Execute other arbitrary statements
- ```?```: Print a help list with all options

To finish a debugging session, either use ```@continue``` to continue and eventually stop at the
next breakpoint or ```@exit``` to skip further breakpoints. After the code has finished, local
variables saved with ```@exfiltrate``` can be accessed in the REPL using the ```safehouse``` variable.

Limitations of using Infiltrator.jl are that local variables cannot be changed, and that it is not
possible to step into further calls or access other function scopes.


## Releasing a new version of Trixi, Trixi2Vtk

- Check whether everything is okay, tests pass etc.
- Set the new version number in `Project.toml` according to the Julian version of semver.
  Commit and push.
- Comment `@JuliaRegistrator register` on the commit setting the version number.
- `JuliaRegistrator` will create a PR with the new version in the General registry.
  Wait for it to be merged.
- Increment the version number in `Project.toml` again with suffix `-pre`. For example,
  if you have released version `v0.2.0`, use `v0.2.1-pre` as new version number.
- When a new version of Trixi was released, check whether the `[compat]` entries
  in `test/Project.toml` in Trixi2Vtk should be updated.
  When a new version of Trixi2Vtk was released, check whether the `[compat]`
  entries in `docs/Project.toml` in Trixi should be updated.

  These entries will also be checked regularly by CompatHelper (once a day). Hence,
  if everything was released correctly, you should only need to do these checks manually
  if new minor versions with changes in the docs of Trixi2Vtk were released
  but no new version of Trixi was released afterwards.



## Preview of the documentation

You can build the documentation of Trixi.jl locally by running
```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs --color=yes docs/make.jl
```
from the Trixi.jl main directory. Then, you can look at the html files generated in
`docs/build`.
For PRs triggered from branches inside the Trixi.jl main repository previews of
the new documentation are generated at `https://trixi-framework.github.io/Trixi.jl/previews/PRXXX`,
where `XXX` is the number of the PR.
This does not work for PRs from forks for security reasons (since anyone could otherwise push
arbitrary stuff to the Trixi website, including malicious code).



## [Developing Trixi2Vtk](@id trixi2vtk-dev)

Trixi2Vtk has Trixi as dependency and uses Trixi's implementation to, e.g., load mesh files.
When developing Trixi2Vtk, one may want to change functions in Trixi to allow them to be reused
in Trixi2Vtk.
To use a locally modified Trixi clone instead of a Trixi release, one can tell Pkg
to use the local source code of Trixi instead of a registered version by running
```julia-repl
(@v1.7) pkg> develop path/to/Trixi.jl
```
