# Unified mesh constructor interface

Trixi.jl provides several mesh types suited for different scenarios.
It may be useful to quickly swap between mesh types
without having to rewrite the mesh construction call.

## Keyword-only interface

All structured mesh types support a keyword-only constructor for rectangular domains
using `refinement_level`, yielding `2^refinement_level` cells per dimension.

```@example mesh-swap
using Trixi

for MeshType in (TreeMesh, StructuredMesh, P4estMesh, T8codeMesh)
    mesh = MeshType(coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                    refinement_level = 2)
    display(mesh)
end
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
