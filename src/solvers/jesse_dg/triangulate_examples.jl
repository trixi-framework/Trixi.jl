# domain with a corner
h = .05
pointlist = [0.0 0.0 ; 1.0 0.0 ; 1.0  1.0 ; 0.6 0.6; 0.0 1.0]'
segmentlist = [1 2 ; 2 3 ; 3 4 ; 4 5 ; 5 1 ]'
segmentmarkerlist = [1, 2, 3, 4, 5]
triin = Triangulate.TriangulateIO()
triin.pointlist = Matrix{Cdouble}(pointlist)
triin.segmentlist = Matrix{Cint}(segmentlist)
triin.segmentmarkerlist = Vector{Int32}(segmentmarkerlist)
triout = triangulate(triin,h^2)

# refinement of a previous mesh
href = h/2
angle = @sprintf("%.15f",20)
area  = @sprintf("%.15f",href^2)
triout2,_ = triangulate("rpa$(area)q$(angle)Q", triout)

# hole domains
h = .05
triin=Triangulate.TriangulateIO()
triin.pointlist=Matrix{Cdouble}([0.0 0.0;
                                 1.0 0.0;
                                 1.0 1.0;
                                 0.0 1.0;
                                 0.4 0.4;
                                 0.6 0.4;
                                 0.6 0.6;
                                 0.4 0.6;
                                 ]')
triin.segmentlist=Matrix{Cint}([1 2; 2 3; 3 4; 4 1; 5 6; 6 7; 7 8; 8 5; ]')
triin.segmentmarkerlist=Vector{Int32}([1, 1,1,1, 2,2,2,2])
triin.holelist=[0.5 0.5]'
triout = triangulate(triin,h^2)
VX,VY,EToV = triangulateIO_to_VXYZEToV(triout)


# plot boundary segments
colors = [:red,:black,:blue,:orange,:green]
xseg = zeros(2,size(triout.segmentlist,2))
yseg = zeros(2,size(triout.segmentlist,2))
segcolor = Symbol[]
for (col,segment) in enumerate(eachcol(triout.segmentlist))
    xseg[:,col] .= VX[segment]
    yseg[:,col] .= VY[segment]
    push!(segcolor,colors[triout.segmentmarkerlist[col]])
end
Plots.plot(xseg,yseg,mark=:circle,color=permutedims(segcolor),leg=false)

rd = RefElemData(Tri(),3)
md = MeshData(VX,VY,EToV,rd)

# find Triangle segment labels of boundary faces
segmentlist = sort(triout.segmentlist,dims=1)
boundary_faces = findall(vec(md.FToF) .== 1:length(md.FToF))
boundary_tag = zeros(Int,length(boundary_faces))
for (f,boundary_face) in enumerate(boundary_faces)
    element = (boundary_face - 1) ÷ rd.Nfaces + 1
    face    = (boundary_face - 1) % rd.Nfaces + 1
    vertex_ids = sort(EToV[element,rd.fv[face]])
    tag_id = findfirst(c->view(segmentlist,:,c)==vertex_ids,axes(segmentlist,2))
    boundary_tag[f] = triout.segmentmarkerlist[tag_id]
end

# bc_tag_dict = Dict(1=>"Dirichlet", 2=>"Neumann", 3=>"Robin", 4=>"Inflow", 5=>"Other")
# Dirichlet() = "Dirichlet bc"
# Neumann() = "Neumann bc"
# Other() = "other bc"
# bcs = (; )

# @unpack x,y,rxJ,sxJ,J = md
# u = @. sin(2*pi*x)*sin(2*pi*y)
# dudx = (rxJ .* (rd.Dr*u) + sxJ.*(rd.Ds*u))./J
# xp,yp = (x->rd.Vp*x).((x,y))
# zz = rd.Vp*dudx
# Plots.scatter(vec(xp),vec(yp),vec(zz),zcolor=vec(zz),leg=false,msw=0,ms=2,cam=(0,90),ratio=1)
