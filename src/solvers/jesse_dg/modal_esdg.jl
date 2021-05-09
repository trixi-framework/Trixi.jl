"""
    function hybridized_SBP_operators(rd::RefElemData{DIMS}) 

Constructs hybridized SBP operators given a RefElemData. Returns operators Qrsth...,VhP,Ph.
"""
function hybridized_SBP_operators(rd::RefElemData{DIMS}) where {DIMS}
    @unpack M,Vq,Pq,Vf,wf,Drst,nrstJ = rd
    Qrst = (D->Pq'*M*D*Pq).(Drst)
    Ef = Vf*Pq
    Brst = (nJ->diagm(wf.*nJ)).(nrstJ)
    Qrsth = ((Q,B)->.5*[Q-Q' Ef'*B;-B*Ef B]).(Qrst,Brst)
    Vh = [Vq;Vf]
    Ph = M\transpose(Vh)
    VhP = Vh*Pq
    return Qrsth...,VhP,Ph
end

function hybridized_SBP_operators(rd::RefElemData{2,Quad})
    Qrh,Qsh,VhP,Ph = invoke(hybridized_SBP_operators,Tuple{RefElemData{2}},rd)
    Qrh,Qsh = sparse.((Qrh,Qsh))
    droptol!(Qrh,50*eps())
    droptol!(Qsh,50*eps())
    return Qrh,Qsh,VhP,Ph
end

function compute_entropy_projection!(Q,rd::RefElemData,cache,eqn)
    @unpack Vq = rd
    @unpack VhP,Ph = cache
    @unpack Uq, VUq, VUh, Uh = cache

    # if this freezes, try 
    #=     CheapThreads.reset_workers!()
           ThreadingUtilities.reinitialize_tasks!()    =#
    StructArrays.foreachfield((uout,u)->matmul!(uout,Vq,u),Uq,Q)
    bmap!(u->cons2entropy(u,eqn),VUq,Uq) # 77.5μs
    StructArrays.foreachfield((uout,u)->matmul!(uout,VhP,u),VUh,VUq)
    bmap!(v->entropy2cons(v,eqn),Uh,VUh) # 327.204 μs

    Nh,Nq = size(VhP)
    Uf = view(Uh,Nq+1:Nh,:) # 24.3 μs

    return Uh,Uf
end

## ======================== 1D rhs codes ==================================

function Trixi.create_cache(mesh::UnstructuredMesh{1}, equations, solver::ModalESDG, RealT, uEltype)

    @unpack VXYZ,EToV = mesh
    md = MeshData(VXYZ...,EToV,rd)
    md = make_periodic(md,rd)

    # make skew symmetric versions of the operators"
    Qrh,VhP,Ph = hybridized_SBP_operators(rd)
    Qrhskew = .5*(Qrh-transpose(Qrh))
    QrhskewTr = typeof(Qrh)(Qrhskew')

    project_and_store! = let Ph=Ph
        (y,x)->mul!(y,Ph,x) # can't use matmul! b/c its applied to a subarray
    end

    # tmp variables for entropy projection
    nvars = nvariables(equations)
    Uq = StructArray{SVector{nvars,Float64}}(ntuple(_->similar(md.xq),nvars))
    VUq = similar(Uq)
    VUh = StructArray{SVector{nvars,Float64}}(ntuple(_->similar([md.xq;md.xf]),nvars))
    Uh = similar(VUh)

    rhse_threads = [similar(Uh[:,1]) for _ in 1:Threads.nthreads()]

    cache = (;md,
            QrhskewTr,VhP,Ph,
            Uq,VUq,VUh,Uh,
            rhse_threads,
            project_and_store!)

    return cache
end

function Trixi.rhs!(dQ, Q::StructArray, t, mesh::UnstructuredMesh{1}, equations,
                    initial_condition, boundary_conditions, source_terms,
                    solver::ModalESDG, cache)

    @unpack md = cache
    @unpack QrhskewTr,VhP,Ph = cache
    @unpack project_and_store! = cache
    @unpack rxJ,J,nxJ,sJ,mapP = md
    rd = solver.rd
    @unpack Vq,wf = rd
    @unpack volume_flux, interface_flux, interface_dissipation = solver

    Nh,Nq = size(VhP)
    skip_index = let Nq=Nq
        (i,j) -> i>Nq && j > Nq
    end

    Uh,Uf = compute_entropy_projection!(Q,rd,cache,equations) # N=2, K=16: 670 μs
        
    @batch for e = 1:md.K 
        rhse = cache.rhse_threads[Threads.threadid()]

        fill!(rhse,zero(eltype(rhse))) # 40ns, (1 allocation: 48 bytes)
        Ue = view(Uh,:,e)    # 8ns (0 allocations: 0 bytes) after @inline in Base.view(StructArray)
        QxTr = LazyArray(@~ @. 2 * rxJ[1,e]*QrhskewTr )

        hadsum_ATr!(rhse, QxTr, volume_flux(1), Ue, skip_index) 
        
        # add in interface flux contributions
        for (i,vol_id) = enumerate(Nq+1:Nh)
            UM, UP = Uf[i,e], Uf[mapP[i,e]]        
            Fx = interface_flux(1)(UP,UM)
            diss = interface_dissipation(SVector{1}(nxJ[i,e]))(UM,UP)
            val = (Fx * nxJ[i,e] + diss*sJ[i,e]) 
            rhse[vol_id] = rhse[vol_id] + val
        end

        # project down and store
        @. rhse = -rhse/J[1,e]

        calc_source_terms(rhse,Ue,t,source_terms,equations,solver,cache,e)
        
        StructArrays.foreachfield(project_and_store!,view(dQ,:,e),rhse) 
    end

    return nothing
end

# ======================= 2D rhs codes =============================

function Trixi.create_cache(mesh::UnstructuredMesh{2}, equations, solver::ModalESDG, RealT, uEltype)

    @unpack VXYZ,EToV = mesh
    md = MeshData(VXYZ...,EToV,rd)
    md = make_periodic(md,rd)

    # for flux differencing on general elements
    Qrh,Qsh,VhP,Ph = hybridized_SBP_operators(rd)
    # make skew symmetric versions of the operators"
    Qrhskew = .5*(Qrh-transpose(Qrh))
    Qshskew = .5*(Qsh-transpose(Qsh))
    QrhskewTr = Matrix(Qrhskew') # punt to dense for now - need rotation?
    QshskewTr = Matrix(Qshskew') 

    project_and_store! = let Ph=Ph
        (y,x)->mul!(y,Ph,x)
    end    

    # tmp variables for entropy projection
    nvars = nvariables(equations)
    Uq = StructArray{SVector{nvars,Float64}}(ntuple(_->similar(md.xq),nvars))
    VUq = similar(Uq)
    VUh = StructArray{SVector{nvars,Float64}}(ntuple(_->similar([md.xq;md.xf]),nvars))
    Uh = similar(VUh)

    # tmp cache for threading
    rhse_threads = [similar(Uh[:,1]) for _ in 1:Threads.nthreads()]

    cache = (;md,
            QrhskewTr,QshskewTr,VhP,Ph,
            Uq,VUq,VUh,Uh,
            rhse_threads,
            project_and_store!)

    return cache
end


function Trixi.rhs!(dQ, Q::StructArray, t, mesh::UnstructuredMesh{2}, equations,
                    initial_condition, boundary_conditions, source_terms,
                    solver::ModalESDG, cache)

    @unpack md,project_and_store! = cache
    @unpack QrhskewTr,QshskewTr,VhP,Ph = cache
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
    @unpack volume_flux, interface_flux, interface_dissipation = solver
    @unpack rd = solver
    @unpack Vq,wf = rd

    Nh,Nq = size(VhP)
    skip_index = let Nq=Nq
        (i,j) -> i > Nq && j > Nq
    end

    Trixi.@timeit_debug Trixi.timer() "compute_entropy_projection!" begin
        Uh,Uf = compute_entropy_projection!(Q,rd,cache,equations) # N=2, K=16: 670 μs
    end

    @batch for e = 1:md.K
        rhse = cache.rhse_threads[Threads.threadid()]

        fill!(rhse,zero(eltype(rhse)))
        Ue = view(Uh,:,e)   
        QxTr = LazyArray(@~ @. 2 *(rxJ[1,e]*QrhskewTr + sxJ[1,e]*QshskewTr))
        QyTr = LazyArray(@~ @. 2 *(ryJ[1,e]*QrhskewTr + syJ[1,e]*QshskewTr))

        hadsum_ATr!(rhse, QxTr, volume_flux(1), Ue, skip_index) 
        hadsum_ATr!(rhse, QyTr, volume_flux(2), Ue, skip_index) 

        for (i,vol_id) = enumerate(Nq+1:Nh)
            UM, UP = Uf[i,e], Uf[mapP[i,e]]
            Fx = interface_flux(1)(UP,UM)
            Fy = interface_flux(2)(UP,UM)
            normal = SVector{2}(nxJ[i,e],nyJ[i,e])/sJ[i,e]
            diss = interface_dissipation(normal)(UM,UP)
            val = (Fx * nxJ[i,e] + Fy * nyJ[i,e] + diss*sJ[i,e]) * wf[i]
            rhse[vol_id] += val
        end

        @. rhse = -rhse / J[1,e]

        # add source terms after scaling by J
        calc_source_terms(rhse,Ue,t,source_terms,equations,solver,cache,e)

        # project down and store
        StructArrays.foreachfield(project_and_store!, view(dQ,:,e), rhse)
    end

    return nothing
end

## ==============================================================================

function calc_source_terms(du::StructArray,u,t,source_terms::Nothing,
                           equations,solver::ModalESDG,cache,element_index)
    return nothing
end

"""
    function calc_source_terms(du::StructArray,u,t,source_terms,
                               equations,solver::ModalESDG,cache,element_index)

Adds source terms on an element (weighted by quadrature weights) prior to projection. 
Independent of dimension or equation.
"""
function calc_source_terms(du::StructArray,u,t,source_terms,
                           equations,solver::ModalESDG,cache,element_index)
    @unpack md = cache
    @unpack rd = solver

    # add source terms to quadrature points
    for i = 1:length(rd.rq)
        du[i] += source_terms(u[i],getindex.(md.xyzq,i,element_index),t,equations)*rd.wq[i]
    end
end