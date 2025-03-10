macro turbo(exprs...)
    body = nothing
    for expr in exprs
        if Meta.isexpr(expr, :for)
            body = expr
        end
    end
    @assert body !== nothing

    function insert_loopinfo!(expr)
        recurse = Meta.isexpr(expr, :for) || Meta.isexpr(expr, :block) ||
                  Meta.isexpr(expr, :let)
        if recurse
            foreach(insert_loopinfo!, expr.args)
        end
        if Meta.isexpr(expr, :for)
            # TODO: Should we insert LLVM loopinfo or `julia.ivdep`?
            push!(expr.args, Expr(:loopinfo, Symbol("julia.simdloop")))
        end
    end
    insert_loopinfo!(body)

    body = Expr(:block,
                Expr(:inbounds, true),
                body,
                Expr(:inbounds, :pop))
    return esc(body)
end
