"""
function plotting_interpolation_matrix(Nplot,rd)

Computes matrix which interpolates from reference interpolation points to equispaced points of degree `Nplot`.
"""
function plotting_interpolation_matrix(Nplot,rd)
    rp,sp = NodesAndModes.equi_nodes(rd.elemShape,Nplot)
    Vp = NodesAndModes.vandermonde(rd.elemShape,rd.N,rp,sp) / rd.VDM
    return Vp
end

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
    triout,_ = triangulate("Q", triin)
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

# inputs = plotting points only. This should be the most general version
# call example: mesh(Vp*u,(x->Vp*x).(rd.rst),(x->Vp*x).(md.xyz))
function Makie.convert_arguments(P::Type{<:Makie.Mesh},uplot,rst_plot,xyz_plot)

    t = permutedims(plotting_triangulation(rst_plot))
    makie_triangles = Makie.to_triangles(t)

    num_elements = size(u_plot,2)
    trimesh = Vector{GeometryBasics.Mesh{3,Float32}}(undef,num_elements)
    coordinates = zeros(length(first(rst_plot)),3)
    for e = 1:num_elements       
        for i = 1:2
            coordinates[:,i] .= view(xyz_plot[i],:,e)
        end
        coordinates[:,3] .= view(uplot,:,e)
        trimesh[e] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates),makie_triangles) # speed this up?
    end
    return tuple(merge([trimesh...]))
end
