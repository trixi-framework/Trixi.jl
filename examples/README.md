# Example elixirs for Trixi

This folder contains example files ("elixirs") that can be used to just try out Trixi and
that also serve as a starting point to create new simulation setups.
Furthermore, these files are also the basis for our automated Trixi tests and
**each new file put in here** should be added to the test sets in
`../test/test_examples_Xd_equation.jl`.

There are also a few files that were configured such that they lend themselves
to doing convergence tests with `convergence_test`, i.e., to determine the
experimental order of convergence (EOC):

* [`2d/elixir_advection_basic.jl`](2d/elixir_advection_basic.jl):
  EOC tests for linear scalar advection with `polydeg = 3`.
* [`2d/elixir_euler_source_terms.jl`](2d/elixir_euler_source_terms.jl):
  EOC tests for Euler equations with `polydeg = 3`.
* [`2d/elixir_mhd_alfven_wave.jl`](2d/elixir_mhd_alfven_wave.jl):
  EOC tests for MHD equations with `polydeg = 3`.
* [`2d/elixir_hypdiff_lax_friedrichs.jl`](2d/elixir_hypdiff_lax_friedrichs.jl):
  EOC tests for hyperbolic diffusion equations with `polydeg = 4`.

Similar setups are available for other spatial dimensions in the directories
`1d` and `3d`.

In general, a good first elixir to try out as a new user is
[`2d/elixir_advection_basic.jl`](2d/elixir_advection_basic.jl),
as it is short, takes less than a second to run, and uses only the basic features
of Trixi.
