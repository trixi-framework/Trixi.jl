# Example elixirs for Trixi.jl

This folder contains example files ("elixirs") that can be used to just try out
Trixi.jl and that also serve as a starting point to create new simulation setups.
Furthermore, these files are also the basis for our automated Trixi.jl tests and
**each new file put in here** should be added to the test sets.

In general, the elixirs are sorted by

- the mesh type,
- the spatial dimension,
- and the discretization type

at the top level. Inside each folder, the elixirs are sorted by the equations,
then the setup, and then, optionally, configuration details. For example,
`elixir_euler_kelvin_helmholtz_instability_amr.jl` indicates an *elixir* for the
compressible *Euler* equations, running the *Kelvin-Helmholtz instability* setup
with adaptive mesh refinement (*AMR*) enabled.

There are also a few files that were configured such that they lend themselves
to doing convergence tests with `convergence_test`, i.e., to determine the
experimental order of convergence (EOC):

* [`tree_2d_dgsem/elixir_advection_basic.jl`](tree_2d_dgsem/elixir_advection_basic.jl):
  EOC tests for linear scalar advection with `polydeg = 3`.
* [`tree_2d_dgsem/elixir_euler_source_terms.jl`](tree_2d_dgsem/elixir_euler_source_terms.jl):
  EOC tests for Euler equations with `polydeg = 3`.
* [`tree_2d_dgsem/elixir_mhd_alfven_wave.jl`](tree_2d_dgsem/elixir_mhd_alfven_wave.jl):
  EOC tests for MHD equations with `polydeg = 3`.
* [`tree_2d_dgsem/elixir_hypdiff_lax_friedrichs.jl`](tree_2d_dgsem/elixir_hypdiff_lax_friedrichs.jl):
  EOC tests for hyperbolic diffusion equations with `polydeg = 4`.

Similar setups are available for other spatial dimensions and for other mesh types.

In general, a good first elixir to try out as a new user is
[`tree_2d_dgsem/elixir_advection_basic.jl`](tree_2d_dgsem/elixir_advection_basic.jl),
as it is short, takes less than a second to run, and uses only the basic features
of Trixi.jl.
