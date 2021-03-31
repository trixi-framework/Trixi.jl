using StartUpDG
using LazyArrays
using BenchmarkTools

N = 4
K = 32
rd = RefElemData(Tri(),N)
VX,VY,EToV = uniform_mesh(Tri(),K)
md = MeshData(VX,VY,EToV,rd)

@unpack Dr,Ds,Vq,Pq = rd
@unpack x,y,rxJ,sxJ,J = md

u1 = @. exp(-10*(x^2+y^2))
u2 = @. exp(-10*(x^2+y^2))
u = [u1,u2]

function Dx(u,rd::RefElemData,md::MeshData)
    @unpack Dr,Ds,Vq,Pq = rd    
    @unpack rxJ,sxJ,J = md
    return (rxJ.*(Dr*u) .+ sxJ.*(Ds*u))./J
end

dudr = similar(u1)
duds = similar(u1)
dudr_local = similar(u1[:,1])
duds_local = similar(u1[:,1])
du = similar.(u)
cache = (;dudr,duds)
cache_local = (;dudr_local,duds_local)

function Dx_prealloc!(du,u,cache,rd::RefElemData,md::MeshData)
    @unpack dudr,duds = cache
    @unpack Dr,Ds,Vq,Pq = rd
    @unpack rxJ,sxJ,J = md
    mul!(dudr,Dr,u)
    mul!(duds,Ds,u)
    @. du = (rxJ*dudr + sxJ*duds)./J
    return nothing
end

function project_naive(u,rd::RefElemData,md::MeshData)
    @unpack Vq,Pq = rd
    @unpack rxJ,sxJ,J = md
    f = Pq*exp.(Vq*u)
    return (rxJ.*(Dr*f) + sxJ.*(Ds*f))./J
end

cache = (; tmp_quad=similar(md.xq), tmp_node=similar(md.x), 
           tmp_rs=(similar(md.x),similar(md.x)))
function project_opt(du,u,cache,rd::RefElemData,md::MeshData)
    @unpack Vq,Pq = rd
    @unpack rxJ,sxJ,J = md
    @unpack tmp_quad, tmp_node, tmp_r, tmp_s = cache

    mul!(tmp_quad,Vq,u)
    @. tmp_quad = exp(tmp_quad)
    mul!(tmp_node,Pq,tmp_quad)

    mul!(tmp_r,Dr,tmp_node)
    mul!(tmp_s,Ds,tmp_node) 
    mul!(du,Ds,tmp_node) 
    for e = 1:size(du,2)
        du_e = @view du[:,e]
        tmp_r_e = @view tmp_r[:,e]
        # tmp_s_e = @view tmp_s[:,e]
        axpby!(rxJ[1,e]/J[1,e],tmp_r_e,sxJ[1,e]/J[1,e],du_e)
        # @. du[:,e] = (rxJ[1,e]* + sxJ[1,e]*tmp_s[:,e])/J[1,e]
    end
    # @. du = (rxJ*tmp_r + sxJ*tmp_s)/J
end

# function Dx_loop!(du,u,cache,rd::RefElemData,md::MeshData) 
#     @unpack Dr,Ds = rd
#     @unpack rxJ,sxJ,J = md
#     @unpack dudr_local,duds_local = cache
#     for e = 1:md.K
#         ue = @view u[:,e]
#         mul!(dudr_local,Dr,ue)
#         mul!(duds_local,Ds,ue)
#         # du[:,e] .= @. (rxJ[:,e]*dudr_local + sxJ[:,e]*duds_local)./J[:,e]
#     end
#     return nothing
# end

# @btime (u->Dx(u,$rd,$md)).($u);
# @btime ((du,u)->Dx_prealloc!(du,u,$cache,$rd,$md)).($du,$u);

# @btime ((du,u)->Dx_loop!(du,u,$cache_local,$rd,$md)).($du,$u);