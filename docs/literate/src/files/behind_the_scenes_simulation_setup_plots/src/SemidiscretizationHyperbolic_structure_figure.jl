using Plots
plot(Shape([(-2.3, 4.5), (2.35, 4.5), (2.35, 2.5), (-2.3, 2.5)]), linecolor = "black",
     fillcolor = "white", label = false, linewidth = 2, size = (800, 600), showaxis = false,
     grid = false, xlim = (-2.4, 2.8), ylim = (-25, 5.5))
annotate!(2.3, 3.5,
          ("SemidiscretizationHyperbolic(mesh, equations, initial_conditions, solver; source_terms,
boundary_conditions, RealT, uEltype, initial_cache)          ", 10, :black, :right))
annotate!(-2.3, 1.5,
          ("creates and returns SemidiscretizationHyperbolic object, initialized using a mesh, equations,
initial_conditions, boundary_conditions, source_terms, solver and cache", 9, :black, :left))
plot!([-1.2, -1.2], [0.6, -2], arrow = true, color = :black, linewidth = 2, label = "")
plot!([-1.2, -1.4], [0.6, -2], arrow = true, color = :black, linewidth = 2, label = "")
plot!([-1.2, -1.0], [0.6, -2], arrow = true, color = :black, linewidth = 2, label = "")
annotate!(-1, -0.7, ("specialized for mesh
and solver types", 9, :black, :left))
plot!([1.25, 1.25], [0.6, -2], arrow = true, color = :black, linewidth = 2, label = "")
plot!([1.25, 1.05], [0.6, -2], arrow = true, color = :black, linewidth = 2, label = "")
plot!([1.25, 1.45], [0.6, -2], arrow = true, color = :black, linewidth = 2, label = "")
annotate!(1.48, -0.7, ("specialized for mesh
and boundary_conditions
types", 9, :black, :left))

plot!(Shape([(-2.3, -2), (-0.1, -2), (-0.1, -4), (-2.3, -4)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)
annotate!(-1.2, -3, ("create_cache(mesh::TreeMesh, equations,
                   solver::Dg, RealT, uEltype)", 10, :black, :center))
plot!([-2.22, -2.22], [-4, -22], arrow = false, color = :black, linewidth = 2, label = "")

plot!(Shape([(-0.05, -2), (2.6, -2), (2.6, -4), (-0.05, -4)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)
annotate!(1.27, -3,
          ("digest_boundary_conditions(boundary_conditions,
                                   mesh, solver, cache)", 10, :black, :center))
annotate!(2.6, -5, ("if necessary, converts passed boundary_conditions
 into a suitable form for processing by Trixi.jl", 9, :black, :right))

plot!(Shape([(-2, -6), (-0.55, -6), (-0.55, -7.1), (-2, -7.1)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)
annotate!(-1.95, -6.5, ("local_leaf_cells(mesh.tree)", 10, :black, :left))
annotate!(-2, -7.5,
          ("returns cells for which an element needs to be created (i.e. all leaf cells)",
           9, :black, :left))
plot!([-2.22, -2], [-6.5, -6.5], arrow = true, color = :black, linewidth = 2, label = "")

plot!(Shape([(-2, -9), (1.73, -9), (1.73, -10.1), (-2, -10.1)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)
annotate!(-1.95, -9.5,
          ("init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)", 10,
           :black, :left))
annotate!(-2, -10.5,
          ("creates and initializes elements, projects Gauss-Lobatto basis onto each of them",
           9, :black, :left))
plot!([-2.22, -2], [-9.5, -9.5], arrow = true, color = :black, linewidth = 2, label = "")

plot!(Shape([(-2, -12), (0.4, -12), (0.4, -13.1), (-2, -13.1)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)
annotate!(-1.95, -12.5,
          ("init_interfaces(leaf_cell_ids, mesh, elements)", 10, :black, :left))
annotate!(-2, -13.5,
          ("creates and initializes interfaces between each pair of adjacent elements of the same size",
           9, :black, :left))
plot!([-2.22, -2], [-12.5, -12.5], arrow = true, color = :black, linewidth = 2, label = "")

plot!(Shape([(-2, -15), (0.5, -15), (0.5, -16.1), (-2, -16.1)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)
annotate!(-1.95, -15.5,
          ("init_boundaries(leaf_cell_ids, mesh, elements)", 10, :black, :left))
annotate!(-2, -17,
          ("creates and initializes boundaries, remembers each boundary element, as well as the coordinates of
each boundary node", 9, :black, :left))
plot!([-2.22, -2], [-15.5, -15.5], arrow = true, color = :black, linewidth = 2, label = "")

plot!(Shape([(-1.6, -18), (1.3, -18), (1.3, -19.1), (-1.6, -19.1)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)
annotate!(-1.55, -18.5,
          ("init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)", 10, :black, :left))
annotate!(-1.6, -20,
          ("creates and initializes mortars (type of interfaces) between each triple of adjacent coarsened
and corresponding small elements", 9, :black, :left))
plot!([-2.22, -1.6], [-18.5, -18.5], arrow = true, color = :black, linewidth = 2,
      label = "")
annotate!(-2.15, -19, ("2D and 3D", 8, :black, :left))

plot!(Shape([(-2, -21), (1.5, -21), (1.5, -23.1), (-2, -23.1)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)
annotate!(-1.95, -22,
          ("create_cache(mesh, equations, dg.volume_integral, dg, uEltype)
for 2D and 3D create_cache(mesh, equations, dg.mortar, uEltype)", 10, :black, :left))
annotate!(-2, -23.5,
          ("add specialized parts of the cache required to compute the volume integral, etc.",
           9, :black, :left))
plot!([-2.22, -2], [-22, -22], arrow = true, color = :black, linewidth = 2, label = "")

savefig("./SemidiscretizationHyperbolic")
