using SparseArrays

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

