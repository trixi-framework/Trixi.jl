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
function plotting_triangulation(rst_plot)

Computes a triangulation of the points in `rst_plot`, which is a tuple containing 
vectors of plotting points on the reference element. Returns a `3 x N_tri` matrix containing
a triangulation of the plotting points, with zero-volume triangles removed.
"""
function plotting_triangulation(rst_plot,tol=50*eps())

    # on-the-fly triangulation of plotting nodes on the reference element
    triin = Triangulate.TriangulateIO()
    triin.pointlist = permutedims(hcat(rst_plot...))
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

"""
    function trixi_pcolor(sol,variable)
    function trixi_pcolor!(sol,variable)    

Plots a pseudocolor plot. Here, sol::TrixiODESolution and variable::Int 
is the index of the solution field to plot.

Other keywords:
- solution_scaling=1.0, # scales the z-values of the solution by this factor
- plot_polydeg = 10     # number of equispaced points used for plotting in each direction 
"""
@Makie.recipe(Trixi_Pcolor, sol, variable) do scene
    # default_theme(scene)...,
    # colormap = theme(scene, :colormap),
    # inspectable = theme(scene, :inspectable), # currently broken
    # colorrange = automatic,
    Attributes(;
        interpolate = false,
        shading = true,
        fxaa = true,
        cycle = [:color => :patchcolor],
        solution_scaling=1.0, # scales the z-values of the solution by this factor
        plot_polydeg = 10     # number of equispaced points used for plotting in each direction 
    )
end

function Makie.plot!(trixi_plot::Trixi_Pcolor{<:Tuple{<:TrixiODESolution, <:Int}})

    variable = trixi_plot[:variable][]
    sol = trixi_plot[:sol][]
    solution_scaling = trixi_plot[:solution_scaling][]

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
    Nplot = trixi_plot[:plot_polydeg][]
    Vp1D = Trixi.polynomial_interpolation_matrix(dg.basis.nodes, LinRange(-1,1,Nplot+1))
    Vp = kron(Vp1D,Vp1D) 
    n_plot_nodes = size(Vp,1)

    # create triangulation for plotting nodes
    rp,sp = (x->Vp*x).((r,s)) # interpolate
    t = permutedims(plotting_triangulation((rp,sp)))
    makie_triangles = Makie.to_triangles(t)

    coordinates_tmp = zeros(Float32,n_nodes,3)
    coordinates = zeros(Float32,n_plot_nodes,3)
    solution_color = zeros(Float32,n_plot_nodes,n_elements)
    trimesh = Vector{GeometryBasics.Mesh{3,Float32}}(undef,n_elements)
    for element in Trixi.eachelement(dg, cache)
        sk = 1
        for j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            u_node = Trixi.get_node_vars(u,semi.equations,dg,i,j,element)
            xy_node = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
            coordinates_tmp[sk,1:2] .= xy_node
            coordinates_tmp[sk,3] = u_node[variable]*solution_scaling
            sk += 1
        end    
        mul!(coordinates,Vp,coordinates_tmp)
        solution_color[:,element] .= @view coordinates[:,3]

        trimesh[element] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates),makie_triangles)
    end
    plotting_mesh = merge([trimesh...])
    Makie.mesh!(trixi_plot, plotting_mesh, color = vec(solution_color))
    trixi_plot
end

"""
    function trixi_wireframe(sol,variable)
    function trixi_wireframe!(sol,variable)    

Plots a wireframe plot of the mesh edges. Here, sol::TrixiODESolution and 
variable::Int is the index of the solution field to plot.

Other keywords:
- solution_scaling=1.0,     # scales the z-values of the solution by this factor
- plot_polydeg = 10         # number of equispaced points used for plotting in each direction 
- color = :black.           # set color = :solution to color by solution
- z_translate_plot = 1.e-3  # translates the wireframe up and down relative to scene axes.

"""
@Makie.recipe(Trixi_Wireframe, sol, variable) do scene
    # default_theme(scene)...,
    # linewidth = theme(scene, :linewidth),
    # colormap = theme(scene, :colormap), # currently broken?
    # inspectable = theme(scene, :inspectable),
    Attributes(;
        linestyle = nothing,
        fxaa = false,
        cycle = [:color],
        color = :black, # set color = :solution to color by solution
        linewidth = 1.0,
        solution_scaling=1.0, # scales the z-values of the solution by this factor
        plot_polydeg = 10,     # number of equispaced points used for plotting in each direction 
        z_translate_plot = 1.e-3
    )
end

function Makie.plot!(trixi_plot::Trixi_Wireframe{<:Tuple{<:TrixiODESolution, <:Int}})

    variable = trixi_plot[:variable][]
    sol = trixi_plot[:sol][]
    semi = sol.prob.p
    @unpack equations, mesh, cache = semi
    dg = semi.solver

    # wrap solution
    u = Trixi.wrap_array(sol.u[end],mesh,equations,dg,cache)
    
    n_nodes_1D = length(dg.basis.nodes)
    n_nodes = n_nodes_1D^2
    n_elements = nelements(dg,cache)

    # reference interpolation operators
    Nplot = trixi_plot[:plot_polydeg][]
    Vp1D = Trixi.polynomial_interpolation_matrix(dg.basis.nodes, LinRange(-1,1,Nplot+1))
    # seems to be the right ordering?
    r1D = dg.basis.nodes
    r = vec([r1D[i] for i = 1:n_nodes_1D, j = 1:n_nodes_1D]) 
    s = vec([r1D[j] for i = 1:n_nodes_1D, j = 1:n_nodes_1D]) 

    # extract local face nodes
    tol = 50*eps()
    e1 = findall(@. abs(s+1)<tol)
    e2 = findall(@. abs(r-1)<tol)
    e3 = findall(@. abs(s-1)<tol)
    e4 = findall(@. abs(r+1)<tol)
    Fmask = hcat(e1,e2,e3,e4)

    x,y,sol_to_plot = ntuple(_->zeros(Float32,n_nodes,n_elements),3)
    for element in Trixi.eachelement(dg, cache)
        sk = 1
        for j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            u_node = Trixi.get_node_vars(u,semi.equations,dg,i,j,element)
            xy_node = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
            x[sk,element] = xy_node[1]
            y[sk,element] = xy_node[2]
            sol_to_plot[sk,element] = u_node[variable]
            sk += 1
        end    
    end

    function face_first_reshape(x,n_nodes_1D,n_nodes,n_elements)
        n_reference_faces = 4 # hardcoded for quads
        xf = view(reshape(x,n_nodes,n_elements),vec(Fmask),:)
        return reshape(xf,n_nodes_1D,n_elements * n_reference_faces)
    end
    xf,yf,sol_f = face_first_reshape.((x,y,sol_to_plot),n_nodes_1D,n_nodes,n_elements)
    xfp,yfp,ufp = map(xf->vec(vcat(Vp1D*xf,fill(NaN,1,size(xf,2)))),(xf,yf,sol_f))

    lw               = trixi_plot[:linewidth][]
    wire_color       = trixi_plot[:color][]
    linestyle        = trixi_plot[:linestyle][]
    solution_scaling = trixi_plot[:solution_scaling][]
    z_translate_plot = trixi_plot[:z_translate_plot][]

    if wire_color==:solution # plot solution 
        Makie.translate!(lines!(trixi_plot,xfp,yfp,ufp*solution_scaling,color=ufp,linewidth=lw,linestyle=linestyle),0,0,z_translate_plot)
    else
        # translate!'s tolerance should be relative...
        Makie.translate!(lines!(trixi_plot,xfp,yfp,ufp*solution_scaling,color=wire_color,linewidth=lw,linestyle=linestyle),0,0,z_translate_plot) 

        # if you want to draw a wireframe under surface as well?
        # Makie.translate!(lines!(trixi_plot,xfp,yfp,ufp,color=wire_color,linewidth=lw,linestyle=linestyle),0,0,-1.e-3) 
    end
    trixi_plot
end



