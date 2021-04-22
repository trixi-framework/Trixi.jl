# Conventions used internally

## Array types and wrapping

To allow adaptive mesh refinement efficiently when using time integrators from
[OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl),
Trixi allows to represent numerical solutions in two different ways. Some discussion
can be found [online](https://github.com/SciML/OrdinaryDiffEq.jl/pull/1275) and
in form of comments describing `Trixi.wrap_array` in the source code of Trixi.
The flexibility introduced by this possible wrapping enables additional
[performance optimizations](https://github.com/trixi-framework/Trixi.jl/pull/509).
However, it comes at the cost of some additional abstractions (and needs to be
used with caution, as described in the source code of Trixi). Thus, we use the
following conventions to distinguish between arrays visible to the time integrator
and wrapped arrays mainly used internally.

- Arrays visible to the time integrator have a suffix `_ode`, e.g., `du_ode`, `u_ode`.
- Wrapped arrays do not have a suffix, e.g., `du, u`.

Methods either accept arrays visible to the time integrator or wrapped arrays
based on the following rules.
- When some solution is passed together with a semidiscretization `semi`, the
  solution must be a `u_ode` that needs to be  wrapped via `wrap_array(u_ode, semi)`
  for further processing.
- When some solution is passed together with the `mesh, equations, solver, cache, ...`,
  it is already wrapped via `wrap_array`.
- Exceptions of this rule are possible, e.g. for AMR, but must be documented in
  the code.
