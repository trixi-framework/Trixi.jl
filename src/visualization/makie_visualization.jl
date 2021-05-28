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
    @unpack solver, mesh = semi

    N = Trixi.polydeg(solver)
    n_nodes_1D = length(solver.basis.nodes)
    n_nodes = n_nodes_1D^2
    n_elements = mesh.n_elements
    pd = PlotData2D(sol) # replace eventually?
    
    Nplot = trixi_plot[:plot_polydeg][]
    Vdm1D = vandermonde(Line(),N,solver.basis.nodes)
    Vp1D = vandermonde(Line(),N,LinRange(-1,1,Nplot+1))/Vdm1D
    Vp = kron(Vp1D,Vp1D) 
    n_plot_nodes = size(Vp,1)

    # build nodes on reference element: seems to be the right ordering?
    r1D = solver.basis.nodes
    r = vec([r1D[i] for i = 1:n_nodes_1D, j = 1:n_nodes_1D]) 
    s = vec([r1D[j] for i = 1:n_nodes_1D, j = 1:n_nodes_1D]) 
    
    sol_to_plot = reshape(pd.data[variable],n_nodes,n_elements)
    x = reshape(pd.x,n_nodes,n_elements)
    y = reshape(pd.y,n_nodes,n_elements)

    rp,sp = (x->Vp*x).((r,s)) # interpolate
    t = permutedims(plotting_triangulation((rp,sp)))
    makie_triangles = Makie.to_triangles(t)

    coordinates = zeros(Float32,n_plot_nodes,3)
    trimesh = Vector{GeometryBasics.Mesh{3,Float32}}(undef,n_elements)
    for element in 1:n_elements    
        xe,ye,ue = view.((x,y,sol_to_plot),:,element)    
        mul!(view(coordinates,:,1),Vp,xe)
        mul!(view(coordinates,:,2),Vp,ye)
        mul!(view(coordinates,:,3),Vp,ue * solution_scaling)
        trimesh[element] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates),makie_triangles)
    end
    plotting_mesh = merge([trimesh...])
    solution_color = vec(Vp*reshape(pd.data[variable],n_nodes,n_elements))
    Makie.mesh!(trixi_plot,plotting_mesh, color = solution_color)
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
    @unpack solver, mesh = semi

    N = Trixi.polydeg(solver)
    n_nodes_1D = length(solver.basis.nodes)
    n_nodes = n_nodes_1D^2
    n_elements = mesh.n_elements
    pd = PlotData2D(sol) # replace eventually?

    Nplot = trixi_plot[:plot_polydeg][]
    Vdm1D = vandermonde(Line(),N,solver.basis.nodes)
    Vp1D = vandermonde(Line(),N,LinRange(-1,1,Nplot+1))/Vdm1D

    # seems to be the right ordering?
    r1D = solver.basis.nodes
    r = vec([r1D[i] for i = 1:n_nodes_1D, j = 1:n_nodes_1D]) 
    s = vec([r1D[j] for i = 1:n_nodes_1D, j = 1:n_nodes_1D]) 

    sol_to_plot = reshape(pd.data[variable],n_nodes,n_elements)
    x = reshape(pd.x,n_nodes,n_elements)
    y = reshape(pd.y,n_nodes,n_elements)

    tol = 50*eps()
    e1 = findall(@. abs(s+1)<tol)
    e2 = findall(@. abs(r-1)<tol)
    e3 = findall(@. abs(s-1)<tol)
    e4 = findall(@. abs(r+1)<tol)
    Fmask = hcat(e1,e2,e3,e4)
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
