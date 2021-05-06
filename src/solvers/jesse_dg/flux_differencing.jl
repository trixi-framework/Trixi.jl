using SparseArrays

# move these out later!
function hybridized_SBP_operators(rd::RefElemData{1})
    @unpack M,Dr,Vq,Pq,Vf,wf,nrJ = rd
    Qr = Pq'*M*Dr*Pq
    Ef = Vf*Pq
    Br = diagm(wf.*nrJ)
    Qrh = .5*[Qr-Qr' Ef'*Br;
            -Br*Ef  Br]
    Vh = [Vq;Vf]
    Ph = M\transpose(Vh)
    VhP = Vh*Pq
    return Qrh,VhP,Ph
end

function hybridized_SBP_operators(rd::RefElemData{2})
    @unpack M,Dr,Ds,Vq,Pq,Vf,wf,nrJ,nsJ = rd
    Qr = Pq'*M*Dr*Pq
    Qs = Pq'*M*Ds*Pq
    Ef = Vf*Pq
    Br = diagm(wf.*nrJ)
    Bs = diagm(wf.*nsJ)
    Qrh = .5*[Qr-Qr' Ef'*Br;
            -Br*Ef  Br]
    Qsh = .5*[Qs-Qs' Ef'*Bs;
            -Bs*Ef  Bs]
    Vh = [Vq;Vf]
    Ph = M\transpose(Vh)
    VhP = Vh*Pq
    return Qrh,Qsh,VhP,Ph
end

function hybridized_SBP_operators(rd::RefElemData{2,Quad})
    Qrh,Qsh,VhP,Ph = invoke(hybridized_SBP_operators,Tuple{RefElemData{2}},rd)
    Qrh, Qsh = sparse.((Qrh,Qsh))
    droptol!(Qrh,50*eps())
    droptol!(Qsh,50*eps())
    return Qrh,Qsh,VhP,Ph
end

# accumulate Q.*F into rhs
function hadsum_ATr!(rhs, ATr, F, u, skip_index=(i,j)->false)
    rows,cols = axes(ATr)
    for i in cols
        ui = u[i]
        val_i = rhs[i]
        for j in rows
            if !skip_index(i,j)
                val_i += ATr[j,i] * F(ui,u[j]) # breaks for tuples, OK for StaticArrays
            end
        end
        rhs[i] = val_i # why not .= here?
    end
end

# accumulate Q.*F into rhs
function hadsum_ATr!(rhs, ATr::AbstractSparseMatrix, F, u)
    rows = rowvals(ATr)
    vals = nonzeros(ATr)
    for i = 1:size(ATr,2) # all ops should be same length
        ui = u[i]
        val_i = rhs[i] # accumulate into existing rhs        
        for row_id in nzrange(ATr,i)
            j = rows[row_id]
            val_i += vals[row_id]*F(ui,u[j])
        end
        rhs[i] = val_i
    end
end


function compute_entropy_projection!(Q,rd::RefElemData,cache,eqn)
    @unpack Vq = rd
    @unpack VhP,Ph = cache
    @unpack Uq, VUq, VUh, Uh = cache

    # if this freezes, try 
    #=     CheapThreads.reset_workers!()
           ThreadingUtilities.reinitialize_tasks!()  =#
    StructArrays.foreachfield((uout,u)->matmul!(uout,Vq,u),Uq,Q)
    bmap!(u->cons2entropy(u,eqn),VUq,Uq) # 77.5μs
    StructArrays.foreachfield((uout,u)->matmul!(uout,VhP,u),VUh,VUq)
    bmap!(v->entropy2cons(v,eqn),Uh,VUh) # 327.204 μs

    Nh,Nq = size(VhP)
    Uf = view(Uh,Nq+1:Nh,:) # 24.3 μs

    return Uh,Uf
end