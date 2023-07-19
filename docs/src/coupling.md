# [Coupling](@id coupling-id)
A complex simulation can consist of different spatial domains in which
different equations are being solved, different numerical methods being used
or the grid structure is different.
One example would be a fluid in a tank and an extended hot plate attached to it.
We would then like to solve the Navier-Stokes equations in the fluid domain
and the heat conduction equations in the plate.
The coupling would happen at the interface through the exchange of thermal energy.

Another type of coupling is bulk or volume coupling.
There we have at least two systems that share all or parts of the domain.
We could, for instance, have a Maxwell system and a fluid system.
The coupling would then occur through the Lorentz force.


## Converter Coupling
We can have the case where the two systems do not share any variables, but
share some of the physics.
Here, the same physics is just represented in a different form and with
different variables.
This is the case for a fluid system on one side and a Vlasov system on the other.
To translate the fields from one description to the other one needs to use
converter functions.

In the general case we have one system with `m` variables `u_i` and another
system with `n` variables `v_j`.
We then define two coupling functions, one that transforms `u_i` into `v_i`
and one that goes the other way.

In their minimal form they take the position vector `x` and state vector `u`
and return the transformed variables.
Examples can be seen in `examples/structured_2d_dgsem/elixir_advection_coupled_converter.jl`
and in `src/coupling_converters/coupling_converters_2d.jl`.
