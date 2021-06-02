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
plotting_interpolation_matrix(dg; kwargs...)

Interpolation matrix which maps discretization nodes to a set of plotting nodes. 
Defaults to the identity matrix of size `length(solver.basis.nodes)`, and interpolates 
to equispaced nodes for DGSEM (set by kwarg `plot_polydeg` in the plotting function). 
"""
plotting_interpolation_matrix(dg; kwargs...) = I(length(dg.basis.nodes)) # is this the right thing for FD-SBP?

function plotting_interpolation_matrix(dg::DGSEM; plot_polydeg = 5)
    return Trixi.polynomial_interpolation_matrix(dg.basis.nodes, LinRange(-1,1,plot_polydeg+1))
end

function generate_plotting_triangulation(sol::TrixiODESolution, variable::Int; plot_polydeg=5)

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
    Vp1D = plotting_interpolation_matrix(dg; plot_polydeg=plot_polydeg)
    Vp = kron(Vp1D,Vp1D) 
    n_plot_nodes = size(Vp,1)

    # create triangulation for plotting nodes
    rp,sp = (x->Vp*x).((r,s)) # interpolate
    t = permutedims(plotting_triangulation((rp,sp)))
    makie_triangles = Makie.to_triangles(t)

    coordinates_tmp = zeros(Float32,n_nodes,3)
    coordinates = zeros(Float32,n_plot_nodes,3)
    # solution_color = zeros(Float32,n_plot_nodes,n_elements)
    trimesh = Vector{GeometryBasics.Mesh{3,Float32}}(undef,n_elements)
    for element in Trixi.eachelement(dg, cache)
        sk = 1
        for j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            u_node = Trixi.get_node_vars(u,semi.equations,dg,i,j,element)
            xy_node = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
            coordinates_tmp[sk,1:2] .= xy_node
            coordinates_tmp[sk,3] = u_node[variable]
            sk += 1
        end    
        mul!(coordinates,Vp,coordinates_tmp)
        # solution_color[:,element] .= @view coordinates[:,3]

        trimesh[element] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates),makie_triangles)
    end
    plotting_mesh = merge([trimesh...])
    return plotting_mesh
end

function generate_plotting_wireframe(sol::TrixiODESolution,variable::Int; plot_polydeg=5)

    semi = sol.prob.p
    @unpack equations, mesh, cache = semi
    dg = semi.solver

    # wrap solution
    u = Trixi.wrap_array(sol.u[end],mesh,equations,dg,cache)
    
    n_nodes_1D = length(dg.basis.nodes)
    n_nodes = n_nodes_1D^2
    n_elements = nelements(dg,cache)

    # reference interpolation operators
    Vp1D = plotting_interpolation_matrix(dg; plot_polydeg = plot_polydeg)   
    
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
    return Makie.Point.(xfp,yfp,ufp)
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
    Attributes(;
        interpolate = false,
        shading = true,
        fxaa = true,
        cycle = [:color => :patchcolor],
        plot_polydeg = 10     # number of equispaced points used for plotting in each direction 
    )
end

function Makie.plot!(trixi_plot::Trixi_Pcolor{<:Tuple{<:TrixiODESolution, <:Int}})

    variable = trixi_plot[:variable][]
    sol = trixi_plot[:sol][]
    plot_polydeg = trixi_plot[:plot_polydeg][]

    plotting_mesh = generate_plotting_triangulation(sol,variable,plot_polydeg=plot_polydeg)

    Makie.mesh!(trixi_plot, plotting_mesh, color = getindex.(plotting_mesh.position,3))
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
    Attributes(;
        linestyle = nothing,
        fxaa = false,
        cycle = [:color],
        color = :black, # set color = :solution to color by solution
        linewidth = 1.0,
        plot_polydeg = 10,     # number of equispaced points used for plotting in each direction 
        z_translate_plot = 1.e-3
    )
end

function Makie.plot!(trixi_plot::Trixi_Wireframe{<:Tuple{<:TrixiODESolution, <:Int}})

    variable = trixi_plot[:variable][]
    sol = trixi_plot[:sol][]
    plot_polydeg = trixi_plot[:plot_polydeg][]

    wire_points = generate_plotting_wireframe(sol,variable,plot_polydeg=plot_polydeg)

    lw               = trixi_plot[:linewidth][]
    wire_color       = trixi_plot[:color][]
    linestyle        = trixi_plot[:linestyle][]
    z_translate_plot = trixi_plot[:z_translate_plot][]

    if wire_color==:solution # plot solution 
        color_option=getindex.(wire_points,3)
    else
        color_option=wire_color
    end
    lineplot = lines!(trixi_plot,wire_points,color=color_option,linewidth=lw,linestyle=linestyle)        
    Makie.translate!(lineplot,0,0,z_translate_plot)

    trixi_plot
end

@Makie.recipe(Trixi_Plot, sol) do scene
    Attributes(;
        fxaa = false,
        cycle = [:color],
        color = :black, # set color = :solution to color by solution
        linewidth = 1.0,
        plot_polydeg = 5,     # number of equispaced points used for plotting in each direction 
        z_translate_plot = 1.e-3
    )
end

function trixi_plot(sol::TrixiODESolution; plot_polydeg=5)

    semi = sol.prob.p
    dg = semi.solver
    @unpack equations, cache, mesh = semi

    fig = Makie.Figure()

    variable_names = Trixi.varnames(cons2cons,equations)
    options = [zip(variable_names,1:length(variable_names))...]
    menu = Makie.Menu(fig, options = options)
    toggle = Makie.Toggle(fig,active=true)
    fig[1, 1] = Makie.vgrid!(
        Makie.Label(fig, "Solution field", width = nothing), menu,
        Makie.Label(fig, "Mesh visible"), toggle;
        tellheight=false,width = 200
    )

    scene = Makie.Axis3(fig[1, 2])

    # interactive variable
    variable = Makie.Node{Int}(1)
    
    # these lines get re-run whenever variable[] is updated
    plotting_mesh = @lift Trixi.generate_plotting_triangulation(sol, $variable, plot_polydeg = plot_polydeg)
    wire_points = @lift Trixi.generate_plotting_wireframe(sol, $variable, plot_polydeg = plot_polydeg)
    Makie.mesh!(scene,plotting_mesh,color=getindex.(plotting_mesh[].position,3),plot_polydeg=plot_polydeg)
    wire_mesh_top = Makie.lines!(scene,wire_points,color=:white) 
    wire_mesh_bottom = Makie.lines!(scene,wire_points,color=:white) 
    Makie.translate!(wire_mesh_top,0,0,1e-3)
    Makie.translate!(wire_mesh_bottom,0,0,-1e-3)
    
    Makie.on(menu.selection) do s
        println("Todo: plot solution field $(variable_names[s])")
        variable[] = s
        Makie.autolimits!(scene)
    end

    menu.selection = 1
    menu.is_open = false 

    # syncs the toggle to the mesh
    Makie.connect!(wire_mesh_top.visible, toggle.active) 
    Makie.connect!(wire_mesh_bottom.visible, toggle.active)
    
    fig
end
