# Style guide
Coding style is an inherently personal - and thus hotly contested - issue. Since code is
usually "written once, read often", it helps regular developers, new users, and reviewers if
code is formatted consistently. We therefore believe in the merit of using a common coding
style throughout Trixi.jl, even at the expense that not everyone can be happy with every
detailed style decision. If you came here because you are furious about our code formatting
rules, here is a happy little whale for you to calm you down: 🐳

## Conventions
The following lists a few coding conventions for Trixi.jl. Note that in addition to these
conventions, we apply and enforce automated source code formatting
(see [below](@ref automated-source-code-formatting) for more details):

  * Modules, types, structs with `CamelCase`.
  * Functions, variables with lowercase `snake_case`.
  * Indentation with 4 spaces (*never* tabs!)
  * Maximum line length (strictly): **92**.
  * Functions that mutate their *input* are named with a trailing `!`.
  * Functions order their parameters [similar to Julia Base](https://docs.julialang.org/en/v1/manual/style-guide/#Write-functions-with-argument-ordering-similar-to-Julia-Base-1).
    * The main modified argument comes first. For example, if the right-hand side `du` is modified,
      it should come first. If only the `cache` is modified, e.g., in `prolong2interfaces!`
      and its siblings, put the `cache` first.
    * Otherwise, use the order `mesh, equations, solver, cache`.
    * If something needs to be specified in more detail for dispatch, put the additional argument before the general one
      that is specified in more detail. For example, we use `have_nonconservative_terms(equations), equations`
      and `dg.mortar, dg`.
  * Prefer `for i in ...` to `for i = ...` for better semantic clarity and greater flexibility.
  * Executable code should only use ASCII characters.
  * Docstrings and comments can and should use Unicode characters where it helps understanding.
  * Multiline expressions should be explicitly grouped by parentheses and not
    rely on Julia's implicit line continuation syntax.
  * When naming multiple functions of a single or similar category, prefer to put the
    *general classification* first and the *specialization* second. Example: Use `flux_central`
    instead of `central_flux`. This helps when searching for available functions on the REPL
    (e.g., when trying to find all flux functions).

## [Automated source code formatting](@id automated-source-code-formatting)
We use [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) to format the
source code of Trixi.jl, which will also enforce *some* of the [Conventions](@ref) listed
above (e.g., line length or indentation with 4 spaces are automatically handled, while
capitalization of names is not). Our format is mostly based on the
[SciML](https://domluna.github.io/JuliaFormatter.jl/stable/sciml_style/)-style formatting
rules. For more details you can have a look at the current
[`.JuliaFormatter.toml`](https://github.com/trixi-framework/Trixi.jl/blob/main/.JuliaFormatter.toml)
file that holds the configuration options we use for JuliaFormatter.jl.

Note that we expect all contributions to Trixi.jl to be formatted with JuliaFormatter.jl
before being merged to the `main` branch. We ensure this by running a automated check on all
PRs that verify that running JuliaFormatter.jl again will not change the source code.

To format your contributions before created a PR (or, at least, before requesting a review
of your PR), you need to install JuliaFormatter.jl first by running
```shell
julia -e 'using Pkg; Pkg.add(PackageSpec(name = "JuliaFormatter", version="1.0.60"))'
```
You can then recursively format the core Julia files in the Trixi.jl repo by executing
```shell
julia -e 'using JuliaFormatter; format(".")'
```
from inside the Trixi.jl repository. For convenience, there is also a script you can
directly run from your terminal shell, which will automatically install JuliaFormatter in a
temporary environment and then run it:
```shell
utils/trixi-format.jl
```
You can get more information about using the convenience script by running it with the
`--help`/`-h` flag.

### Checking formatting before committing
It can be convenient to check the formatting of source code automatically before each commit.
We use git-hooks for it and provide a `pre-commit` script in the `utils` folder. The script uses
[JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) just like formatting script that
runs over the whole Trixi.jl directory.
You can copy the `pre-commit`-script into `.git/hooks/pre-commit` and it will check your formatting
before each commit. If errors are found the commit is aborted and you can add the corrections via
```shell
git add -p
```
