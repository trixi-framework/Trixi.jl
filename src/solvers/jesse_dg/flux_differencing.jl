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
    # make skew symmetric versions of the operators"
    Qrhskew = .5*(Qrh-transpose(Qrh))
    return Qrhskew,VhP,Ph
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

    # make skew symmetric versions of the operators"
    Qrhskew = .5*(Qrh-transpose(Qrh))
    Qshskew = .5*(Qsh-transpose(Qsh))
    return Qrhskew,Qshskew,VhP,Ph
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