using StartUpDG
using Plots
using Printf
using Triangulate
import Triangulate:triangulate
import StartUpDG:MeshData

# convenience routine 
function Triangulate.triangulate(triin::TriangulateIO,maxarea,minangle=20)
    angle = @sprintf("%.15f",minangle)
    area  = @sprintf("%.15f",maxarea)
    triout,_ = triangulate("pa$(area)q$(angle)Q", triin)
    return triout
end

function triangulateIO_to_VXYZEToV(triout::TriangulateIO)
    VX,VY = (triout.pointlist[i,:] for i = 1:size(triout.pointlist,1))
    EToV = permutedims(triout.trianglelist)
    Base.swapcols!(EToV,2,3) # to match MeshData ordering
    return VX,VY,EToV
end

function plotMesh(VX,VY,EToV)
    xmesh = Float64[]
    ymesh = Float64[]
    for vertex_ids in eachrow(EToV)
        ids = vcat(vertex_ids, vertex_ids[1])
        append!(xmesh,[VX[ids];NaN])
        append!(ymesh,[VY[ids];NaN])
    end
    display(Plots.plot(xmesh,ymesh,linecolor=:black,legend=false,ratio=1))
end

function plotMesh!(VX,VY,EToV)
    xmesh = Float64[]
    ymesh = Float64[]
    for vertex_ids in eachrow(EToV)
        ids = vcat(vertex_ids, vertex_ids[1])
        append!(xmesh,[VX[ids];NaN])
        append!(ymesh,[VY[ids];NaN])
    end
    display(Plots.plot!(xmesh,ymesh,linecolor=:black,legend=false,ratio=1))
end

