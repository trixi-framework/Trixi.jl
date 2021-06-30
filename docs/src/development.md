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
To enter the package REPL mode, press `]` in the standard Julia REPL mode. Then, execute
```julia-repl
(@v1.6) pkg> add Revise
```
Now you are able to run Trixi from the REPL, change Trixi code between runs,
**and** enjoy the advantages of the compilation cache! Before you start using
Revise regularly, please be aware of some of the [Pitfalls when using Revise](@ref).

Another recommended package for working from the REPL is
[OhMyREPL.jl](https://github.com/KristofferC/OhMyREPL.jl). It can be installed
by running
```julia-repl
(@v1.6) pkg> add OhMyREPL
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
applies should you opt to install the postprocessing tools
[Trixi2Vtk](https://github.com/trixi-framework/Trixi2Vtk.jl) and
[Trixi2Img](https://github.com/trixi-framework/Trixi2Img.jl) manually such that
you can modify their implementations.


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
  flux_ranocha(u_ll, u_rr, orientation, equations::CompressibleEulerEquations1D) in Trixi at ~/.julia/dev/Trixi/src/equations/1d/compressible_euler.jl:416
  flux_ranocha(u_ll, u_rr, orientation, equations::CompressibleEulerEquations2D) in Trixi at ~/.julia/dev/Trixi/src/equations/2d/compressible_euler.jl:865
  flux_ranocha(u_ll, u_rr, orientation, equations::CompressibleEulerEquations3D) in Trixi at ~/.julia/dev/Trixi/src/equations/3d/compressible_euler.jl:710
  ```
- Use `methodswith` to discover methods associated to a given type etc.
  ```julia-repl
  julia> methodswith(CompressibleEulerEquations2D)
  [1] initial_condition_convergence_test(x, t, equations::CompressibleEulerEquations2D) in Trixi at ~/.julia/dev/Trixi/src/equations/2d/compressible_euler.jl:38
  [...]
  ```
- Use `@which` (or `@edit`) for method calls.
  ```julia-repl
  julia> @which trixi_include(default_example())
  trixi_include(elixir::AbstractString; kwargs...) in Trixi at ~/.julia/dev/Trixi/src/run.jl:72
  ```
- Use `apropos` to search through the documentation and docstrings.
  ```julia-repl
  julia> apropos("MHD")
  Trixi.initial_condition_constant
  Trixi.initial_condition_rotor
  Trixi.IdealGlmMhdEquations2D
  Trixi.initial_condition_jeans_instability
  Trixi.IdealGlmMhdEquations3D
  Trixi.flux_derigs_etal
  Trixi.initial_condition_weak_blast_wave
  Trixi.initial_condition_blast_wave
  Trixi.initial_condition_convergence_test
  Trixi.noncons_interface_flux
  Trixi.initial_condition_orszag_tang
  Trixi.calc_fast_wavespeed_roe
  Trixi.flux_hll
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



## Releasing a new version of Trixi, Trixi2Vtk, Trixi2Img

- Check whether everything is okay, tests pass etc.
- Set the new version number in `Project.toml` according to the Julian version of semver.
  Commit and push.
- Comment `@JuliaRegistrator register` on the commit setting the version number.
- `JuliaRegistrator` will create a PR with the new version in the General registry.
  Wait for it to be merged.
- Increment the version number in `Project.toml` again with suffix `-pre`. For example,
  if you have released version `v0.2.0`, use `v0.2.1-pre` as new version number.
- Set the correct version number in the badge "GitHub commits since tagged version"
  in README.md.
  The badge will only show up correctly if TagBot has released a new version. This will
  be done automatically.
- When a new version of Trixi was released, check whether the `[compat]` entries
  in `test/Project.toml` in Trixi2Vtk/Trixi2Img should be updated.
  When a new version of Trixi2Vtk/Trixi2Img was released, check whether the `[compat]`
  entries in `docs/Project.toml` in Trixi should be updated.

  These entries will also be checked regularly by CompatHelper (once a day). Hence,
  if everything was released correctly, you should only need to do these checks manually
  if new minor versions with changes in the docs of Trixi2Vtk/Trixi2Img were released
  but no new version of Trixi was released afterwards.



## Preview of the documentation

You can build the documentation of Trixi.jl locally by running
```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```
from the Trixi.jl main directory. Then, you can look at the html files generated in
`docs/build`.
For PRs triggered from branches inside the Trixi.jl main repository previews of
the new documentation are generated at `https://trixi-framework.github.io/Trixi.jl/previews/PRXXX`,
where `XXX` is the number of the PR.
This does not work for PRs from forks for security reasons (since anyone could otherwise push
arbitrary stuff to the Trixi website, including malicious code).



## Developing Trixi2Vtk (@id trixi2vtk-dev)

Trixi2Vtk has Trixi as dependency and uses Trixi's implementation to, e.g., load mesh files.
When developing Trixi2Vtk, one may want to change functions in Trixi to allow them to be reused
in Trixi2Vtk.
To use a locally modified Trixi clone instead of a Trixi release, one can tell Pkg
to use the local source code of Trixi instead of a registered version by running
```julia-repl
(@v1.6) pkg> develop path/to/Trixi.jl
```
