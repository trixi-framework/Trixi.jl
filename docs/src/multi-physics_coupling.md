# [Multi-physics coupling](@id multi-physics-coupling)
A complex simulation can consist of different spatial domains in which
different equations are being solved, different numerical methods being used
or the grid structure is different.
One example would be a fluid in a tank and an extended hot plate attached to it.
We would then like to solve the Navier-Stokes equations in the fluid domain
and the heat conduction equations in the plate.
The coupling would happen at the interface through the exchange of thermal energy.


## Converter Coupling
We can have the case where the two systems do not share any variables, but
share some of the physics.
Here, the same physics is just represented in a different form and with
different variables.
This is the case for a fluid system on one side and a Vlasov system on the other.
To translate the fields from one description to the other one needs to use
converter functions.

In the general case, we have a system A with $m$ variables $u_{A,i}$ and another
system B with $n$ variables $u_{B,j}$.
We then define two coupling functions, one that transforms $u_A$ into $u_B$
and one that goes the other way.

In their minimal form they take the position vector `x` and state vector `u`
and return the transformed variables.
Examples can be seen in `examples/structured_2d_dgsem/elixir_advection_coupled.jl`.


## Warning about Binary Compatibility
Currently the coordinate values on the nodes can differ by machine precision when
simulating the mesh and when splitting the mesh in multiple domains.
This is an issue coming from the coordinate interpolation on the nodes.
As a result, running a simulation in a single system and in two coupled domains
may result in a difference of the order of the machine precision.
While this is not an issue for most practical problems, it is best to keep this in mind when comparing test runs.

