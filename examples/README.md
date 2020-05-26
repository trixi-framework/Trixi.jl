# Example parameter files for Trixi
This folder contains parameter files that can be used to just try out Trixi and
that also serve as a starting point to create new simulation setups.
Furthermore, these files are also the basis for our automated Trixi tests and
**each new file put in here** should be added either to the basic or the extended test
set in [`../test/test_examples.jl`](../test/test_examples.jl).

There are also a few files that were configured such that they lend themselves
to doing convergence tests with `convtest`, i.e., to determine the experimental order
of convergence (EOC):

* [`parameters.toml`](parameters.toml):  
  EOC tests for linear scalar advection with `N = 3`.
* [`parameters_source_terms.toml`](parameters_source_terms.toml):  
  EOC tests for Euler equations with `N = 3`.
* [`parameters_alfven_wave.toml`](parameters_alfven_wave.toml):  
  EOC tests for MHD equations with `N = 3`.

In general, a good first parameter file to try out as a new user is
[`parameters.toml`](parameters.toml), as it is short, takes less than a second
to run, and uses only the basic features of Trixi.
