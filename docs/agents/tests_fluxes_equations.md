# Tests for numerical fluxes, wave speeds, and equations

This file complements "Adding and changing tests" in
[`AGENTS.md`](../../AGENTS.md). Read it before adding or changing a numerical
flux, a wave speed estimate, an equation, or pointwise functions of an
equation.

- When adding a numerical flux or wave speed estimate, add unit tests to
  `test/test_unit.jl` (see the existing `"Unit: Consistency check for ..."`
  and `"Unit: Equivalent ..."` test items). For a conservative numerical flux
  `f`, check consistency with the physical flux, i.e.,
  `f(u, u, orientation, equations) ≈ flux(u, orientation, equations)` and the
  same with `normal_direction`. For all fluxes and wave speed estimates with
  `normal_direction` methods, check agreement with the `orientation` methods
  for axis-aligned unit normals. For new methods with a `normal_direction`
  (including new equations), add the equations and fluxes to
  `normal_direction_test_cases` in `test/test_unit.jl`; the test items
  `"Unit: Wave speed estimates with normal_direction"` and
  `"Unit: Numerical fluxes with normal_direction"` fail otherwise.
- Except for consistency checks, unit tests for numerical fluxes and wave speed
  estimates should use *different* left and right states: many copy-paste bugs
  are invisible for `u_ll == u_rr`.
- When adding an equation or new pointwise functions, add type stability tests
  for `Float32` and `Float64` to `test/test_type.jl` (see the existing
  `"Type stability: ..."` test items using `@inferred`).
