# [Multi-physics coupling](@id multi-physics-coupling)
A complex simulation can consist of different spatial domains in which
different equations are being solved, different numerical methods being used
or the grid structure is different.
One example would be a fluid in a tank and an extended hot plate attached to it.
We would then like to solve the Navier-Stokes equations in the fluid domain
and the heat conduction equations in the plate.
The coupling would happen at the interface through the exchange of thermal energy.


## Converter coupling
It may happen that the two systems to be coupled do not share any variables, but
share some of the physics.
In such a situation, the same physics is just represented in a different form and with
a different set of variables.
This is the case, for instance assuming two domains, if there is a fluid system in one domain
and a Vlasov system in the other domain.
In that case we would have variables representing distribution functions of
the Vlasov system on one side and variables representing the mechanical quantities, like density,
of the fluid system.
To translate the fields from one description to the other one needs to use
converter functions.
These functions need to be hand tailored by the user in the elixir file where each
pair of coupled systems requires two coupling functions, one for each direction.

In the general case, we have a system $A$ with $m$ variables
$u_{A,i}, \: i = 1, \dots, m$ and another
system $B$ with $n$ variables $u_{B,j}, \: j = 1, \dots, n$.
We then define two coupling functions, one that transforms $u_A$ into $u_B$
and one that goes the other way.

In their minimal form they take the position vector $x$, state vector $u$
and the equations of the two coupled systems
and return the transformed variables.
By passing the equations we can make use of their parameters, if they are required.
Examples can be seen in `examples/structured_2d_dgsem/elixir_advection_coupled.jl`.


## GlmSpeedCallback for coupled MHD simulations

When simulating an MHD system and the [`GlmSpeedCallback`](@ref) is required,
we need to specify for which semidiscretization we need the GLM speed updated.
This can be done with an additional parameter called `semi_indices`, which
is a tuple containing the semidiscretization indices for all systems
that require the GLM speed updated.

An example elixir can be found at [`examples/structured_2d_dgsem/elixir_mhd_coupled.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/structured_2d_dgsem/elixir_mhd_coupled.jl).


## Warning about binary compatibility
Currently the coordinate values on the nodes can differ by machine precision when
simulating the mesh and when splitting the mesh in multiple domains.
This is an issue coming from the coordinate interpolation on the nodes.
As a result, running a simulation in a single system and in two coupled domains
may result in a difference of the order of the machine precision.
While this is not an issue for most practical problems, it is best to keep this in mind when comparing test runs.

