# Unified mesh constructor interface

Trixi.jl provides several mesh types suited for different scenarios.
It may be useful to quickly swap between mesh types
without having to rewrite the mesh construction call.

An example demonstrating mesh-type swapping for the same simulation setup is
provided in `examples/special_elixirs/elixir_advection_mesh_swap.jl`.

## Constructor overview

All structured mesh types support three equivalent styles for rectangular domains:

### Style 1 — `cells_per_dimension` positional (classic)

```julia
StructuredMesh((16, 16), (-1.0, -1.0), (1.0, 1.0))
P4estMesh((16, 16), (-1.0, -1.0), (1.0, 1.0); polydeg = 3)
T8codeMesh((16, 16), (-1.0, -1.0), (1.0, 1.0))
DGMultiMesh(dg, (16, 16);
            coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0))
```

### Style 2 — `initial_refinement_level` (like `TreeMesh`)

Directly mirrors the [`TreeMesh`](@ref) interface, yielding `2^initial_refinement_level`
cells per dimension:

```julia
# Original TreeMesh call
mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0);
                initial_refinement_level = 4,
                n_cells_max = 30_000)

# Drop-in replacements — only n_cells_max needs to be removed:
mesh = StructuredMesh((-1.0, -1.0), (1.0, 1.0); initial_refinement_level = 4)
mesh = P4estMesh((-1.0, -1.0), (1.0, 1.0); initial_refinement_level = 4, polydeg = 3)
mesh = T8codeMesh((-1.0, -1.0), (1.0, 1.0); initial_refinement_level = 4)
mesh = DGMultiMesh(dg; initial_refinement_level = 4,
                   coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0))
```

Note: for `StructuredMesh` and `DGMultiMesh`, `initial_refinement_level` directly sets
`cells_per_dimension = 2^initial_refinement_level`. For `P4estMesh` and `T8codeMesh`,
a single tree per dimension is created and refined `initial_refinement_level` times,
which also yields `2^initial_refinement_level` leaf cells per dimension.

### Style 3 — keyword-based (like `P4estMesh`)

All parameters as keywords, same style as the original [`P4estMesh`](@ref) interface:

```julia
# Original P4estMesh keyword call
mesh = P4estMesh((16, 16); polydeg = 3,
                 coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0))

# Equivalent calls on other mesh types:
mesh = StructuredMesh((16, 16);
                      coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0))
mesh = T8codeMesh((16, 16);
                  coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0))
mesh = DGMultiMesh(dg; cells_per_dimension = (16, 16),
                   coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0))
```

## Mapping and face-based constructors

For curvilinear meshes, `mapping` and `faces` can also be passed positionally,
matching the [`StructuredMesh`](@ref) interface:

```julia
# All three are equivalent:
StructuredMesh((16, 16), mapping)
P4estMesh((16, 16), mapping; polydeg = 3)
T8codeMesh((16, 16), mapping)

# Face-based:
StructuredMesh((16, 16), faces)
P4estMesh((16, 16), faces; polydeg = 3)
T8codeMesh((16, 16), faces)
```

## Notes on `TreeMesh`

[`TreeMesh`](@ref) has a fundamentally different design and cannot be used as a
drop-in target in all cases:

- It requires `n_cells_max` with no equivalent in other mesh types.
- It only supports refinement where all dimensions have the same number of cells (`2^level`).
- It only supports rectangular domains — no `mapping` or `faces`.

When swapping **from** `TreeMesh` to another type, remove `n_cells_max` and
choose `initial_refinement_level` or compute `cells_per_dimension = (2^level, 2^level, ...)` manually.

When swapping **to** `TreeMesh` from another type, the swap is only possible
if the domain is rectangular and `cells_per_dimension` is a power of two with the same value in all dimensions.

## `trees_per_dimension` vs. `cells_per_dimension`

In [`P4estMesh`](@ref) and [`T8codeMesh`](@ref), the first positional argument is named
`trees_per_dimension`, reflecting the p4est concept of a *forest of trees*: the argument
specifies how many trees exist per dimension, each of which may be further refined
by `initial_refinement_level`. The total leaf-cell count per dimension is therefore
`trees_per_dimension * 2^initial_refinement_level`.

In contrast, [`StructuredMesh`](@ref) and [`DGMultiMesh`](@ref) use `cells_per_dimension`
for the final cell count with no further refinement.

The Style 1 convenience constructors (`P4estMesh(cells_per_dimension, coordinates_min, ...)`)
pass `cells_per_dimension` directly as `trees_per_dimension` with `initial_refinement_level = 0`
by default, so `cells_per_dimension` equals the actual leaf-cell count per dimension.

The Style 2 convenience constructors (`P4estMesh(coordinates_min, coordinates_max; initial_refinement_level=...)`)
use `trees_per_dimension = (1, 1, ...)` internally and forward `initial_refinement_level`,
so the leaf-cell count per dimension is `2^initial_refinement_level` — consistent with `TreeMesh`.
