using MAT
using Setfield
using NodesAndModes
using LinearAlgebra
using UnPack
using StartUpDG

include("flux_differencing.jl")

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

        # convert Hicken format to 
        r = @. 2*r-1 
        s = @. 2*s-1
        w = 2.0 * w/sum(w)
    else
        error("No nodes found for N=$N with quadrature_strength = $quadrature_strength")
    end

    quad_rule_face = gauss_lobatto_quad(0,0,N+1) # hardcoded
    return (r,s,w),quad_rule_face 
end

struct DiagESummationByParts{DIM,Tv,Ti}
    points::NTuple{DIM,Vector{Tv}}
    M::Diagonal{Tv,Vector{Tv}}
    Mf::Diagonal{Tv,Vector{Tv}}
    Qrst::NTuple{DIM,Matrix{Tv}}
    E_face_extraction::Matrix{Tv}
    Fmask::Matrix{Ti}
end

function DiagESummationByParts(elementType::Tri, N, quad_rule_vol, quad_rule_face)
    
    # build polynomial reference element using quad rules
    rd_sbp = RefElemData(elementType, N; quad_rule_vol=quad_rule_vol, quad_rule_face=quad_rule_face)

    # determine Fmask = indices of face nodes among volume nodes
    @unpack wq,wf,rq,sq,rf,sf,Nfaces = rd_sbp   
    rf,sf = (x->reshape(x,length(rf)÷Nfaces,Nfaces)).((rf,sf))
    Fmask = zeros(Int,length(rf)÷Nfaces,Nfaces) # 
    E_face_extraction = zeros(length(rf),length(rq)) # extraction matrix
    for i in eachindex(rq)
        for f = 1:rd_sbp.Nfaces
            tol = 1e-14
            id = findall(@. abs(rq[i]-rf[:,f]) + abs(sq[i]-sf[:,f]) .< tol)
            Fmask[id,f] .= i
            E_face_extraction[id .+ (f-1)*size(rf,1),i] .= 1
        end
    end

    # build traditional SBP operators from hybridized operators. 
    Qrh,Qsh,VhP,Ph = hybridized_SBP_operators(rd_sbp)

    # Reference: "High-order entropy stable dG methods for the SWE: curved triangular meshes and GPU acceleration" 
    # Section 3.2 of https://arxiv.org/pdf/2005.02516.pdf
    # https://doi.org/10.1016/j.camwa.2020.11.006   
    Vh_sbp = [I(length(rq)); E_face_extraction]
    Qr_sbp = Vh_sbp'*Qrh*Vh_sbp
    Qs_sbp = Vh_sbp'*Qsh*Vh_sbp

    M = Diagonal(wq)
    Mf = Diagonal(wf)
    DiagESummationByParts((rq,sq),M,Mf,(Qr_sbp,Qs_sbp),E_face_extraction,Fmask)
end

"Entropy stable solver using nodal (collocated) DG methods"
struct NodalESDG{DIM,ElemType,F1,F2,F3,Tv,Ti}
    rd::RefElemData{DIM,ElemType} # polynomial base used to construct the SBP op
    sbp_operators::DiagESummationByParts{DIM,Tv,Ti} # non-polynomial SBP operators
    volume_flux::F1 
    interface_flux::F2
    interface_dissipation::F3
end

function NodalESDG(N,elementType,r,s,w,
                   trixi_volume_flux::F1,
                   trixi_interface_flux::F2,
                   trixi_interface_dissipation::F3,
                   equations) where {F1,F2,F3}


    
    volume_flux, interface_flux, interface_dissipation = let equations=equations
        volume_flux(orientation) = (u_ll,u_rr)->trixi_volume_flux(u_ll,u_rr,orientation,equations)
        interface_flux(orientation) = (u_ll,u_rr)->trixi_interface_flux(u_ll,u_rr,orientation,equations)
        interface_dissipation(orientation) = (u_ll,u_rr)->trixi_interface_dissipation(u_ll,u_rr,orientation,equations)
        volume_flux,interface_flux,interface_dissipation
    end

    NodalESDG(rd_sbp,volume_flux,interface_flux,interface_dissipation)
end


