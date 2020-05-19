# Style guide
The following lists a few coding conventions for Trixi:

  * Modules, types, structs with `CamelCase`
  * Functions, variables with lowercase `snake_case`
  * Indentation with 2 spaces (*never* tabs!), line continuations indented with 4 spaces
  * Maximum line length (strictly): **100**
  * Prefer `for i in ...` to `for i = ...` for better semantic clarity and greater flexibility
  * Executable code should only use ASCII characters
  * Docstrings and comments can and should use Unicode characters where it helps understanding


Based on that, and personal experience, a formatting tool with a few helpful
options is included in `utils/julia-format.jl`. Note, however, that this tool is
not yet optimal, as it re-indents too greedily.

This is a list of handy style guides that are mostly consistent with each
other and this guide, and which have been used as a basis:

  * [https://www.juliaopt.org/JuMP.jl/stable/style/](https://www.juliaopt.org/JuMP.jl/stable/style/)
  * [https://github.com/jrevels/YASGuide](https://github.com/jrevels/YASGuide)

