"""
    struct ModalESDG{N,DIM,ElemType,F1,F2,F3} 
        volume_flux::F1 
        interface_flux::F2
        interface_dissipation::F3
        cons2entropy::F4
        entropy2cons::F5
    end

    `volume_flux`,`interface_flux`,`interface_dissipation` expect arguments `flux(orientation)(u_ll,u_rr)`. 
    A convenience constructor using Trixi's flux functions is provided. 
"""
struct ModalESDG{DIM,ElemType,Tv,F1,F2,F3,F4,F5} 
    rd::RefElemData{DIM,ElemType,Tv}
    volume_flux::F1 
    interface_flux::F2
    interface_dissipation::F3
    cons2entropy::F4
    entropy2cons::F5
end

"""
    function ModalESDG(rd::RefElemData,
        trixi_volume_flux::F1,
        trixi_interface_flux::F2,
        trixi_interface_dissipation::F3,
        cons2entropy::F4,
        entropy2cons::F5,        
        equations) where {F1,F2,F3}
    
Initialize a ModalESDG solver with Trixi fluxes as arguments, where trixi_*_flux has the form of
    trixi_*_flux(u_ll,u_rr,orientation,equations)

"""
function ModalESDG(rd::RefElemData,
                   trixi_volume_flux::F1,
                   trixi_interface_flux::F2,
                   trixi_interface_dissipation::F3,
                   cons2entropy::F4,
                   entropy2cons::F5,
                   equations) where {F1,F2,F3,F4,F5}

    volume_flux, interface_flux, interface_dissipation = let equations=equations
        volume_flux(orientation) = (u_ll,u_rr)->trixi_volume_flux(u_ll,u_rr,orientation,equations)
        interface_flux(orientation) = (u_ll,u_rr)->trixi_interface_flux(u_ll,u_rr,orientation,equations)
        interface_dissipation(orientation) = (u_ll,u_rr)->trixi_interface_dissipation(u_ll,u_rr,orientation,equations)
        volume_flux,interface_flux,interface_dissipation
    end
    return ModalESDG(rd,volume_flux,interface_flux,interface_dissipation,cons2entropy,entropy2cons)
end

function Base.show(io::IO, solver::ModalESDG{DIM}) where {DIM}
    println("Modal ESDG solver in $DIM dimension with ")
    println("   volume flux           = $(solver.volume_flux.trixi_volume_flux)")
    println("   interface flux        = $(solver.interface_flux.trixi_interface_flux)")    
    println("   interface dissipation = $(solver.interface_dissipation.trixi_interface_dissipation)")        
    println("   cons2entropy          = $(solver.cons2entropy)")            
    println("   entropy2cons          = $(solver.entropy2cons)")                
end

Base.real(solver::ModalESDG) = Float64 # is this for DiffEq.jl?
Trixi.ndofs(md::MeshData, solver::ModalESDG, cache) = length(solver.rd.r)*md.K

