# Unified mesh constructor interface

Trixi.jl provides several mesh types suited for different scenarios.
It may be useful to quickly swap between mesh types
without having to rewrite the mesh construction call.

## Keyword-only interface

All structured mesh types support a keyword-only constructor for rectangular domains
using `refinement_level`, yielding `2^refinement_level` cells per dimension.

```@example mesh-swap
using Trixi

mesh = TreeMesh(coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                refinement_level = 2, n_cells_max = 30_000)
```

```@example mesh-swap
mesh = StructuredMesh(coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                      refinement_level = 2)
```

```@example mesh-swap
mesh = P4estMesh(coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                 refinement_level = 2)
```

```@example mesh-swap
mesh = T8codeMesh(coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                  refinement_level = 2)
```

[`DGMultiMesh`](@ref) also supports the same keyword arguments, but requires a solver `dg`
as the first positional argument:

```@example mesh-swap
dg = DGMulti(polydeg = 1, element_type = Quad())
mesh = DGMultiMesh(dg; coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                   refinement_level = 2)
```

Note: for `StructuredMesh` and `DGMultiMesh`, `refinement_level` directly sets
`cells_per_dimension = 2^refinement_level`. For `P4estMesh` and `T8codeMesh`,
a single tree per dimension is created and refined `refinement_level` times,
which also yields `2^refinement_level` leaf cells per dimension.

## Notes on `TreeMesh`

[`TreeMesh`](@ref) has a different design and cannot be used as a
drop-in target in all cases:

- It requires `n_cells_max` with no equivalent in other mesh types.
- It only supports refinement where all dimensions have the same number of cells (`2^level`).
- It only supports rectangular domains — no `mapping` or `faces`.

When swapping **from** `TreeMesh` to another type, remove `n_cells_max`.
When swapping **to** `TreeMesh` from another type, add `n_cells_max` and ensure
the domain is rectangular.
