"""
function compute_triangle_area(tri)

Computes the area of a triangle given `tri`, which is a tuple of three points (vectors).
Formula from https://en.wikipedia.org/wiki/Shoelace_formula
"""
function compute_triangle_area(tri)
    A,B,C = tri
    return .5*(A[1]*(B[2] - C[2]) + B[1]*(C[2]-A[2]) + C[1]*(A[2]-B[2]))
end

"""
    function plotting_triangulation(reference_plotting_coordinates)

Computes a triangulation of the points in `reference_plotting_coordinates`, which is a tuple containing
vectors of plotting points on the reference element (e.g., reference_plotting_coordinates = (r,s)). The
reference element is assumed to be [-1,1]^d.

Returns `t` which is a `3 x N_tri` Matrix{Int} containing indices of triangles in the triangulation of
the plotting points, with zero-volume triangles removed.

For example, r[t[1,i]] returns the first reference coordinate of the 1st point on the ith triangle.
"""
function plotting_triangulation(reference_plotting_coordinates,tol=50*eps())

    # on-the-fly triangulation of plotting nodes on the reference element
    triin = Triangulate.TriangulateIO()
    triin.pointlist = permutedims(hcat(reference_plotting_coordinates...))
    triout,_ = Triangulate.triangulate("Q", triin)
    t = triout.trianglelist

    # filter out sliver triangles
    has_volume = fill(true, size(t,2))
    for i in axes(t, 2)
        ids = @view t[:, i]
        x_points = @view triout.pointlist[1, ids]
        y_points = @view triout.pointlist[2, ids]
        area = compute_triangle_area(zip(x_points, y_points))
        if abs(area) < tol
            has_volume[i] = false
        end
    end
    return t[:,findall(has_volume)]
end

"""
    function plotting_interpolation_matrix(dg; kwargs...)

Interpolation matrix which maps discretization nodes to a set of plotting nodes.
Defaults to the identity matrix of size `length(solver.basis.nodes)`, and interpolates
to equispaced nodes for DGSEM (set by kwarg `nvisnodes` in the plotting function).

Example:
```julia
A = plotting_interpolation_matrix(dg)
A*dg.basis.nodes # => vector of nodes at which to plot the solution
```
"""
# note: we cannot use UniformScaling to define the interpolation matrix since we use it with `kron`
# to define a multi-dimensional interpolation matrix later.
plotting_interpolation_matrix(dg; kwargs...) = I(length(dg.basis.nodes))

function plotting_interpolation_matrix(dg::DGSEM; nvisnodes = 2*length(dg.basis.nodes))
    return Trixi.polynomial_interpolation_matrix(dg.basis.nodes, LinRange(-1,1,nvisnodes))
end

"""
    function generate_plotting_triangulation(sol::TrixiODESolution, variable_to_plot::Int; nvisnodes=5)

Generates a plotting triangulation which can be used by Makie's `mesh` function.

Arguments:
- sol: TrixiODESolution
- variable_to_plot: index of solution field to plot

Example:
```julia
tri = Trixi.generate_plotting_triangulation(sol,1)
Makie.mesh(tri,color=getindex.(tri.position,3))
```
The color is specified to be the z-value of the solution, which there does not seem to be a default solution for yet.
"""
function generate_plotting_triangulation(sol::TrixiODESolution, variable_to_plot::Int; nvisnodes=5)

    semi = sol.prob.p
    dg = semi.solver
    @unpack equations, cache, mesh = semi

    # make solution
    u = sol.u[end]
    u = Trixi.wrap_array(u, mesh, equations, dg, cache)

    n_nodes_1D = length(dg.basis.nodes)
    n_nodes = n_nodes_1D^2
    n_elements = nelements(dg, cache)

    # build nodes on reference element (seems to be the right ordering)
    r1D = dg.basis.nodes
    r = vec([r1D[i] for i = 1:n_nodes_1D, j = 1:n_nodes_1D])
    s = vec([r1D[j] for i = 1:n_nodes_1D, j = 1:n_nodes_1D])

    # reference plotting nodes
    Vp1D = plotting_interpolation_matrix(dg; nvisnodes=nvisnodes)
    Vp = kron(Vp1D, Vp1D)
    n_plot_nodes = size(Vp, 1)

    # create triangulation for plotting nodes
    rp,sp = (x->Vp*x).((r, s)) # interpolate dg nodes to plotting nodes

    # construct a triangulation of the plotting nodes
    t = permutedims(plotting_triangulation((rp, sp)))
    makie_triangles = Makie.to_triangles(t)

    # trimesh[i] holds the GeometryBasics.Mesh containing solution plotting
    # information on the ith element.
    trimesh = Vector{GeometryBasics.Mesh{3,Float32}}(undef, n_elements)
    coordinates_tmp = zeros(Float32, n_nodes, 3)
    coordinates = zeros(Float32, n_plot_nodes, 3)
    for element in Trixi.eachelement(dg, cache)

        # extract x,y coordinates and solutions on each element
        sk = 1
        for j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            u_node = Trixi.get_node_vars(u,semi.equations, dg, i, j, element)
            xy_node = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
            coordinates_tmp[sk,1:2] .= xy_node
            coordinates_tmp[sk,3] = u_node[variable_to_plot]
            sk += 1
        end

        # interpolates both xy coordinates and solution values to plotting points
        mul!(coordinates, Vp, coordinates_tmp)

        trimesh[element] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates), makie_triangles)
    end

    # merge meshes on each element into one large mesh
    plotting_mesh = merge([trimesh...])
    return plotting_mesh
end

function generate_plotting_wireframe(sol::TrixiODESolution, variable_to_plot::Int; nvisnodes=5)

    semi = sol.prob.p
    @unpack equations, mesh, cache = semi
    dg = semi.solver

    n_nodes_1D = length(dg.basis.nodes)
    n_nodes = n_nodes_1D^2
    n_elements = nelements(dg,cache)

    # reference interpolation operators
    Vp1D = plotting_interpolation_matrix(dg; nvisnodes = nvisnodes)

    # reconstruct reference nodes (assumes 2D ordering on reference quad is x_i,y_j with i first).
    r1D = dg.basis.nodes
    r = vec([r1D[i] for i = 1:n_nodes_1D, j = 1:n_nodes_1D])
    s = vec([r1D[j] for i = 1:n_nodes_1D, j = 1:n_nodes_1D])

    # extract indices of local face nodes
    tol = 50*eps()
    face_1 = findall(@. abs(s+1) < tol)
    face_2 = findall(@. abs(r-1) < tol)
    face_3 = findall(@. abs(s-1) < tol)
    face_4 = findall(@. abs(r+1) < tol)
    Fmask = hcat(face_1,face_2,face_3,face_4)

    # need to call `wrap_array` before using get_node_vars, etc...
    u = Trixi.wrap_array(sol.u[end], mesh, equations, dg, cache)

    x,y,sol_to_plot = ntuple(_ -> zeros(Float32, n_nodes, n_elements), 3)
    for element in Trixi.eachelement(dg, cache)
        sk = 1
        for j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            u_node = Trixi.get_node_vars(u, semi.equations, dg, i, j, element)
            xy_node = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
            x[sk,element] = xy_node[1]
            y[sk,element] = xy_node[2]
            sol_to_plot[sk, element] = u_node[variable_to_plot]
            sk += 1
        end
    end

    # These 5 lines extract the face values on each element from the arrays x,y,sol_to_plot.
    # The resulting arrays are then reshaped so that xf,yf,sol_f are Matrix types of size
    # (Number of face plotting nodes) x (Number of faces).
    function face_first_reshape(x, n_nodes_1D, n_nodes, n_elements)
        n_reference_faces = 4 # hardcoded for quads
        xf = view(reshape(x, n_nodes, n_elements), vec(Fmask), :)
        return reshape(xf, n_nodes_1D, n_elements * n_reference_faces)
    end
    xf, yf, sol_f = face_first_reshape.((x, y, sol_to_plot), n_nodes_1D, n_nodes, n_elements)

    # this line does three things:
    # - interpolates face nodal points to plotting points
    # - use NaNs to break up lines corresponding to each face
    # - converts the final array to a vector
    xfp, yfp, ufp = map(xf->vec(vcat(Vp1D*xf, fill(NaN, 1, size(xf, 2)))), (xf, yf, sol_f))

    return Makie.Point.(xfp, yfp, ufp)
end

function trixi_plot(sol::TrixiODESolution; variable_to_plot_in = 1, nvisnodes=5)

    semi = sol.prob.p
    dg = semi.solver
    @unpack equations, cache, mesh = semi

    fig = Makie.Figure()

    # set up menu and toggle switch
    variable_names = Trixi.varnames(cons2cons,equations)
    options = [zip(variable_names,1:length(variable_names))...]
    menu = Makie.Menu(fig, options = options)
    toggle_sol_mesh = Makie.Toggle(fig,active=true)
    toggle_mesh = Makie.Toggle(fig,active=true)
    fig[1, 1] = Makie.vgrid!(
        Makie.Label(fig, "Solution field", width = nothing), menu,
        Makie.Label(fig, "Solution mesh visible"), toggle_sol_mesh,
        Makie.Label(fig, "Mesh visible"), toggle_mesh;
        tellheight=false, width = 200
    )

    ax = Makie.LScene(fig[1,2],scenekw = (show_axis = false,))

    # interactive menu variable_to_plot
    menu.selection[] = variable_to_plot_in
    menu.i_selected[] = variable_to_plot_in # initialize a menu

    # these lines get re-run whenever variable_to_plot[] is updated
    plotting_mesh = Makie.@lift(Trixi.generate_plotting_triangulation(sol, $(menu.selection), nvisnodes = nvisnodes))
    solution_z = Makie.@lift(getindex.($plotting_mesh.position, 3))
    Makie.mesh!(ax, plotting_mesh, color=solution_z, nvisnodes=nvisnodes)

    # mesh overlay
    wire_points = Makie.@lift(Trixi.generate_plotting_wireframe(sol, $(menu.selection), nvisnodes = nvisnodes))
    wire_mesh_top = Makie.lines!(ax, wire_points, color=:white)
    wire_mesh_bottom = Makie.lines!(ax, wire_points, color=:white)
    Makie.translate!(wire_mesh_top, 0, 0, 1e-3)
    Makie.translate!(wire_mesh_bottom, 0, 0, -1e-3)

    # mesh below the solution
    function compute_z_offset(solution_z)
        zmin = minimum(solution_z)
        zrange = (x->x[2]-x[1])(extrema(solution_z))
        return zmin - .25*zrange
    end
    z_offset = Makie.@lift(compute_z_offset($solution_z))
    get_flat_points(wire_points, z_offset) = [Makie.Point(point.data[1:2]..., z_offset) for point in wire_points]
    flat_wire_points = Makie.@lift get_flat_points($wire_points, $z_offset)
    wire_mesh_flat = Makie.lines!(ax, flat_wire_points, color=:black)

    # reset colorbar each time solution changes
    Makie.Colorbar(fig[1, 3], limits = Makie.@lift(extrema($solution_z)), colormap = :viridis, flipaxis = false)

    # syncs the toggle to the mesh
    Makie.connect!(wire_mesh_top.visible, toggle_sol_mesh.active)
    Makie.connect!(wire_mesh_bottom.visible, toggle_sol_mesh.active)
    Makie.connect!(wire_mesh_flat.visible, toggle_mesh.active)

    # typing this pulls up the figure (similar to display(plot!()) in Plots.jl)
    fig
end
