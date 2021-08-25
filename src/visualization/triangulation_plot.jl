using Trixi, Triangulate, Plots, TriplotRecipes

# The following methods are based on PR#613 by Jesse Chan. Instead of using Makie.jl this uses Plots.jl.
function triangulation_plot(sol, variable_to_plot::Int; nvisnodes=5, solution_variables=nothing)

    semi = sol.prob.p
    dg = semi.solver
    @unpack equations, cache, mesh = semi

    # make solution
    u = sol.u[end]
    u = Trixi.wrap_array(u,mesh,equations,dg,cache)

    n_nodes_1D = length(dg.basis.nodes)
    n_nodes = n_nodes_1D^2
    n_elements = nelements(dg,cache)

    # build nodes on reference element (seems to be the right ordering)
    r1D = dg.basis.nodes
    r = vec([r1D[i] for i = 1:n_nodes_1D, j = 1:n_nodes_1D])
    s = vec([r1D[j] for i = 1:n_nodes_1D, j = 1:n_nodes_1D])

    # reference plotting nodes
    Vp1D = plotting_interpolation_matrix(dg; nvisnodes=nvisnodes)
    Vp = kron(Vp1D,Vp1D)
    n_plot_nodes = size(Vp,1)

    # create triangulation for plotting nodes
    rp,sp = (x->Vp*x).((r,s)) # interpolate dg nodes to plotting nodes

    # construct a triangulation of the plotting nodes
    t = permutedims(plotting_triangulation((rp,sp)))

    coordinates_tmp = zeros(Float32,n_nodes,3,n_elements)
    coordinates = zeros(Float32,n_plot_nodes,3,n_elements)
    for element in Trixi.eachelement(dg, cache)
        # extract x,y coordinates and solutions on each element
        sk = 1
        for j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            u_node = Trixi.get_node_vars(u,semi.equations,dg,i,j,element)
            xy_node = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
            coordinates_tmp[sk,1:2,element] .= xy_node
            coordinates_tmp[sk,3,element] = u_node[variable_to_plot]
            sk += 1
        end

        # interpolates both xy coordinates and solution values to plotting points
        coordinates[:,:,element] = Vp*coordinates_tmp[:,:,element]
    end

    # Assemble the element-wise triangulations to a global one.
    coordinates_out = zeros(n_plot_nodes*n_elements, 3)
    n_tri = size(t,1)
    t_out = zeros(n_tri*n_elements, 3)
    for element in 0:n_elements-1
        coordinates_out[(1:n_plot_nodes).+(element*n_plot_nodes),:] = coordinates[:,:,element+1]
        t_out[(1:n_tri).+(element*n_tri),1:3] = t.+n_plot_nodes*element
    end

    # Extract data for plotting.
    solution_variables_ = Trixi.digest_solution_variables(equations, solution_variables)
    variable_names = SVector(Trixi.varnames(solution_variables_, equations))
    x = coordinates_out[:,1]
    y = reverse(coordinates_out[:,2])
    z = coordinates_out[:,3]

    plot(aspect_ratio=:equal, size=(800,811), xguide="x", yguide="y", title=variable_names[variable_to_plot])
    tripcolor!(x,y,z,convert(Matrix{Int64}, t_out'))
end

plotting_interpolation_matrix(dg; kwargs...) = I(length(dg.basis.nodes)) # is this the right thing for FD-SBP?

function plotting_interpolation_matrix(dg::DGSEM; nvisnodes = 2*length(dg.basis.nodes))
    return Trixi.polynomial_interpolation_matrix(dg.basis.nodes, LinRange(-1,1,nvisnodes))
end

function plotting_triangulation(reference_plotting_coordinates,tol=50*eps())

    # on-the-fly triangulation of plotting nodes on the reference element
    triin = Triangulate.TriangulateIO()
    triin.pointlist = permutedims(hcat(reference_plotting_coordinates...))
    triout,_ = Triangulate.triangulate("Q", triin)
    t = triout.trianglelist

    # filter out sliver triangles
    has_volume = fill(true,size(t,2))
    for i in axes(t,2)
        ids = @view t[:,i]
        x_points = @view triout.pointlist[1,ids]
        y_points = @view triout.pointlist[2,ids]
        area = compute_triangle_area(zip(x_points,y_points))
        if abs(area) < tol
            has_volume[i] = false
        end
    end
    return t[:,findall(has_volume)]
end

function compute_triangle_area(tri)
    A,B,C = tri
    return .5*(A[1]*(B[2] - C[2]) + B[1]*(C[2]-A[2]) + C[1]*(A[2]-B[2]))
end
