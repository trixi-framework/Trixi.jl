Julia-based 1D DGSEM Extreme
============================


## Style guide
The following lists a few conventions that have been used so far:

*   Modules, types, structs with `CamelCase`
*   Functions, variables with lowercase `snake_case`
*   Indentation with 2 spaces (never tabs!), line continuations indented with 4
    spaces
*   Maximum line length: 100
*   Prefer `for i in 1:n` to `for i = 1:n` for better semantic clarity

Based on that, and personal experience, a formatting tool with a few helpful
options is included in `utils/julia-format.jl`. Note, however, that this tool is
not yet optimal, as it re-indents too greedily.

This is a list of handy style guides that are mostly consistent with each
other and this guide:

*   https://www.juliaopt.org/JuMP.jl/stable/style/
*   https://github.com/jrevels/YASGuide

