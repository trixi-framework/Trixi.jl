function build_hSBP_ops(rd::RefElemData{2})
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