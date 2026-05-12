# Unified mesh constructor interface

Trixi.jl provides several mesh types suited for different scenarios.
It may be useful to quickly swap between mesh types
without having to rewrite the mesh construction call.

An example demonstrating mesh-type swapping for the same simulation setup is
provided in `examples/special_elixirs/elixir_advection_mesh_swap.jl`.

## Keyword-only interface

All structured mesh types support a keyword-only constructor for rectangular domains
using `initial_refinement_level`. This directly mirrors the [`TreeMesh`](@ref) interface,
yielding `2^initial_refinement_level` cells per dimension:

```julia
# TreeMesh (reference call)
mesh = TreeMesh(coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                initial_refinement_level = 4, n_cells_max = 30_000)

# Drop-in replacements. Only n_cells_max needs to be removed:
mesh = StructuredMesh(coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                      initial_refinement_level = 4)
mesh = P4estMesh(coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                 initial_refinement_level = 4)
mesh = T8codeMesh(coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                  initial_refinement_level = 4)
```

[`DGMultiMesh`](@ref) also supports the same keyword arguments, but requires a solver `dg`
as the first positional argument:

```julia
mesh = DGMultiMesh(dg; coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                   initial_refinement_level = 4)
```

Note: for `StructuredMesh` and `DGMultiMesh`, `initial_refinement_level` directly sets
`cells_per_dimension = 2^initial_refinement_level`. For `P4estMesh` and `T8codeMesh`,
a single tree per dimension is created and refined `initial_refinement_level` times,
which also yields `2^initial_refinement_level` leaf cells per dimension.

## Notes on `TreeMesh`

[`TreeMesh`](@ref) has a fundamentally different design and cannot be used as a
drop-in target in all cases:

- It requires `n_cells_max` with no equivalent in other mesh types.
- It only supports refinement where all dimensions have the same number of cells (`2^level`).
- It only supports rectangular domains — no `mapping` or `faces`.

When swapping **from** `TreeMesh` to another type, remove `n_cells_max`.
When swapping **to** `TreeMesh` from another type, add `n_cells_max` and ensure
the domain is rectangular.

