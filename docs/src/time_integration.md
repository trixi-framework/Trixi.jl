# Time integration methods

Trixi is compatible with the [SciML ecosystem for ordinary differential equations](https://diffeq.sciml.ai/latest/).
In particular, explicit Runge-Kutta from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
are tested extensively.
Interesting classes of time integration methods are
- [Explicit low-storage Runge-Kutta time integration](https://diffeq.sciml.ai/latest/solvers/ode_solve/#Explicit-Strong-Stability-Preserving-Runge-Kutta-Methods-for-Hyperbolic-PDEs-(Conservation-Laws))
- [Strong stability preserving methods](https://diffeq.sciml.ai/latest/solvers/ode_solve/#Low-Storage-Methods)

!!! note

  If you use explicit Runge-Kutta from [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl),
  the total number of `rhs!` calls can be (slightly) bigger than the number of steps times the number
  of stages, e.g. to allow for interpolation (dense output), root-finding for continuous callbacks,
  and error-based time step control. In general, you often should not need to worry about this if you
  use Trixi.
