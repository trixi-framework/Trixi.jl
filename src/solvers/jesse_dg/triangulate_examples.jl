function demo_domain(h = .05)
    # domain with a corner
    pointlist = [0.0 0.0 ; 1.0 0.0 ; 1.0  1.0 ; 0.6 0.6; 0.0 1.0]'
    segmentlist = [1 2 ; 2 3 ; 3 4 ; 4 5 ; 5 1 ]'
    segmentmarkerlist = [1, 2, 3, 4, 5]
    triin = Triangulate.TriangulateIO()
    triin.pointlist = Matrix{Cdouble}(pointlist)
    triin.segmentlist = Matrix{Cint}(segmentlist)
    triin.segmentmarkerlist = Vector{Int32}(segmentmarkerlist)
    triout = triangulate(triin,h^2)
    return triout
end

# refinement of a previous mesh given the current mesh size h
function refine(triout, h, href = h/2)
    angle = @sprintf("%.15f",20)
    area  = @sprintf("%.15f",href^2)
    triout2,_ = triangulate("rpa$(area)q$(angle)Q", triout)
    returnt triout2
end

# domain with a square hole
function square_hole_domain(h = .05)
    # hole domains
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
    return triout
end

# scramjet domain 
function scramjet(h = .1)
    # hole domains
    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}([0.0 0.0;
                                     8.0 0.0;
                                     8.0 0.8;    
                                     0.0 2.0;
                                     2.0 0.7;
                                     4.0 0.2; 
                                     7.0 0.6;
                                     6.0 0.7;
                                    ]')
    triin.segmentlist=Matrix{Cint}([1 2; 2 3; 3 4; 4 1; 5 6; 6 7; 7 8; 8 5; ]')
    # 1 = wall, 2 = inflow, 3 = outflow 
    triin.segmentmarkerlist=Vector{Int32}([1, 3, 1, 2, 1, 1, 1, 1])
    hole_x = sum(triin.pointlist[1,5:8])/length(triin.pointlist[1,5:8])
    hole_y = sum(triin.pointlist[2,5:8])/length(triin.pointlist[2,5:8])
    triin.holelist=[hole_x hole_y]'
    triout = triangulate(triin,h^2)
    return triout
end


VX,VY,EToV = triangulateIO_to_VXYZEToV(triout)

# plot boundary segments as different colors
function plot_segment_tags(triout::TriangulateIO)    
    tags = unique(triout.segmentmarkerlist)
    num_colors = length(tags)
    colors = range(HSV(0,1,1), stop=HSV(360-360÷num_colors,1,1), length=num_colors)
    xseg = zeros(2,size(triout.segmentlist,2))
    yseg = zeros(2,size(triout.segmentlist,2))
    segcolor = HSV{Float32}[]
    for (col,segment) in enumerate(eachcol(triout.segmentlist))
        xseg[:,col] .= triout.pointlist[1,segment]
        yseg[:,col] .= triout.pointlist[2,segment]
        push!(segcolor,colors[triout.segmentmarkerlist[col]])
    end
    Plots.plot()
    for i = 1:num_colors
        color_ids = findall(triout.segmentmarkerlist .== tags[i])

        # hack to get around issues with multiple legend labels appearing when plotting multiple series
        x_i = vec([xseg[:,color_ids]; fill(NaN,length(color_ids))']) 
        y_i = vec([yseg[:,color_ids]; fill(NaN,length(color_ids))']) 

        Plots.plot!(x_i,y_i,mark=:circle,color=permutedims(segcolor[color_ids]),
                    ratio = 1,label=string(tags[i])) 
    end
    display(plot!())
end

rd = RefElemData(Tri(),3)
md = MeshData(VX,VY,EToV,rd)



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
