using MAT # convert from .mat files
using Setfield 
using LinearAlgebra
using UnPack
using NodesAndModes
using StartUpDG

include("ModalESDG.jl")
include("modal_esdg.jl")

"""
    struct NodalESDG{ElemType,DIM,F1,F2,F3,Tv,Ti}
        sbp_operators::DiagESummationByParts{ElemType,DIM,Tv,Ti} # non-polynomial SBP operators
        rd::RefElemData{DIM,ElemType}
        volume_flux::F1 
        interface_flux::F2
        interface_dissipation::F3
    end
Entropy stable solver using nodal (collocated) DG methods
"""
struct NodalESDG{ElemType,DIM,F1,F2,F3,Tv,Ti}
    sbp_operators::DiagESummationByParts{ElemType,DIM,Tv,Ti} # non-polynomial SBP operators
    rd::RefElemData{DIM,ElemType}
    volume_flux::F1 
    interface_flux::F2
    interface_dissipation::F3
end

function NodalESDG(N,elementType,
                   trixi_volume_flux::F1,
                   trixi_interface_flux::F2,
                   trixi_interface_dissipation::F3,
                   equations; quadrature_strength=2*N-1) where {F1,F2,F3}
    
    volume_flux, interface_flux, interface_dissipation = let equations=equations
        volume_flux(orientation) = (u_ll,u_rr)->trixi_volume_flux(u_ll,u_rr,orientation,equations)
        interface_flux(orientation) = (u_ll,u_rr)->trixi_interface_flux(u_ll,u_rr,orientation,equations)
        interface_dissipation(orientation) = (u_ll,u_rr)->trixi_interface_dissipation(u_ll,u_rr,orientation,equations)
        volume_flux,interface_flux,interface_dissipation
    end

    quad_rule_vol,quad_rule_face = diagE_sbp_nodes(elementType, N; quadrature_strength=quadrature_strength)
    rd_sbp = RefElemData(elementType,N; quad_rule_vol = quad_rule_vol, quad_rule_face = quad_rule_face)
    sbp_ops = DiagESummationByParts(elementType, N, rd_sbp)

    NodalESDG(sbp_ops,rd_sbp,volume_flux,interface_flux,interface_dissipation)
end


parsevec(type, str) = str |>
  (x -> split(x, ", ")) |>
  (x -> map(y -> parse(type, y), x))
  
"Triangular SBP nodes with diagonal boundary matrices. Nodes from "
function diagE_sbp_nodes(elem::Tri, N; quadrature_strength=2*N-1)
    if quadrature_strength==2*N-1
        # from Ethan Kubatko, private communication
        vars = matread("./sbp_nodes/KubatkoQuadratureRules.mat"); 
        rs = vars["Q_GaussLobatto"][N]["Points"]
        r,s = (rs[:,i] for i = 1:size(rs,2))
        w = vec(vars["Q_GaussLobatto"][N]["Weights"])
    elseif quadrature_strength==2*N
        # from Jason Hicken https://github.com/OptimalDesignLab/SummationByParts.jl/tree/work
        lines = readlines("sbp_nodes/tri_diage_p$N.dat") 
        r = parsevec(Float64,lines[11])
        s = parsevec(Float64,lines[12])
        w = parsevec(Float64,lines[13])

        # convert Hicken format to biunit right triangle
        r = @. 2*r-1 
        s = @. 2*s-1
        w = 2.0 * w/sum(w)
    else
        error("No nodes found for N=$N with quadrature_strength = $quadrature_strength")
    end

    quad_rule_face = gauss_lobatto_quad(0,0,N+1) # hardcoded
    return (r,s,w),quad_rule_face 
end

struct DiagESummationByParts{T<:AbstractElemShape,DIM,Tv,Ti}
    elementType::T
    points::NTuple{DIM,Vector{Tv}} # sbp nodes
    wq::Vector{Tv} # volume weights
    wf::Vector{Tv} # face weights
    Qrst::NTuple{DIM,Matrix{Tv}} # differentiation operators
    Ef::Matrix{Tv} # face node extraction operator/"interpolation" to face nodes
    Fmask::Vector{Ti} 
end

function DiagESummationByParts(elementType::Tri, N; quadrature_strength = 2*N-1)
    quad_rule_vol, quad_rule_face = diagE_sbp_nodes(elementType, N; quadrature_strength=quadrature_strength)

    # build polynomial reference element using quad rules
    rd_sbp = RefElemData(elementType, N; quad_rule_vol=quad_rule_vol, quad_rule_face=quad_rule_face)

    return DiagESummationByParts(elementType, N, rd_sbp)
end
    
function DiagESummationByParts(elementType::Tri, N, rd_sbp::RefElemData)
    
    # determine Fmask = indices of face nodes among volume nodes
    @unpack wq,wf,rq,sq,rf,sf,Nfaces = rd_sbp   
    rf,sf = (x->reshape(x,length(rf)÷Nfaces,Nfaces)).((rf,sf))
    Fmask = zeros(Int,length(rf)÷Nfaces,Nfaces) # 
    Ef = zeros(length(rf),length(rq)) # extraction matrix
    for i in eachindex(rq)
        for f = 1:rd_sbp.Nfaces
            tol = 1e-14
            id = findall(@. abs(rq[i]-rf[:,f]) + abs(sq[i]-sf[:,f]) .< tol)
            Fmask[id,f] .= i
            Ef[id .+ (f-1)*size(rf,1),i] .= 1
        end
    end

    # build traditional SBP operators from hybridized operators. 
    Qrh,Qsh,VhP,Ph = hybridized_SBP_operators(rd_sbp)

    # See Section 3.2 of [High-order entropy stable dG methods for the SWE](https://arxiv.org/pdf/2005.02516.pdf) by Wu and Chan 2021.
    # [DOI](https://doi.org/10.1016/j.camwa.2020.11.006)
    Vh_sbp = [I(length(rq)); Ef]
    Qr_sbp = Vh_sbp'*Qrh*Vh_sbp
    Qs_sbp = Vh_sbp'*Qsh*Vh_sbp

    # Br = Qr_sbp + Qr_sbp'
    # Bs = Qs_sbp + Qs_sbp'

    return DiagESummationByParts(elementType,(rq,sq),wq,wf,(Qr_sbp,Qs_sbp),Ef,vec(Fmask))
end


Base.real(solver::NodalESDG) = Float64 # is this for DiffEq.jl?
Trixi.ndofs(mesh::UnstructuredMesh, solver::NodalESDG, cache) = length(solver.rd.rq)*cache.md.K

