# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function refine_element!(u::AbstractArray{<:Any, 2}, element_id,
                         old_u, old_element_id,
                         adaptor::Nothing, equations, solver::FV)
    # Store new element ids
    lower_left_id = element_id
    lower_right_id = element_id + 1
    upper_left_id = element_id + 2
    upper_right_id = element_id + 3

    @boundscheck begin
        @assert old_element_id >= 1
        @assert size(old_u, 1) == nvariables(equations)
        @assert size(old_u, 2) >= old_element_id
        @assert element_id >= 1
        @assert size(u, 1) == nvariables(equations)
        @assert size(u, 2) >= element_id + 3
    end

    u_local = get_node_vars(old_u, equations, solver, old_element_id)
    set_node_vars!(u, u_local, equations, solver, lower_left_id)
    set_node_vars!(u, u_local, equations, solver, lower_right_id)
    set_node_vars!(u, u_local, equations, solver, upper_left_id)
    set_node_vars!(u, u_local, equations, solver, upper_right_id)

    return nothing
end

function coarsen_elements!(u::AbstractArray{<:Any, 2}, element_id,
                           old_u, old_element_id,
                           adaptor::Nothing, equations::AbstractEquations{2},
                           solver::FV)
    # Store old element ids
    lower_left_id = old_element_id
    lower_right_id = old_element_id + 1
    upper_left_id = old_element_id + 2
    upper_right_id = old_element_id + 3

    @boundscheck begin
        @assert old_element_id >= 1
        @assert size(old_u, 1) == nvariables(equations)
        @assert size(old_u, 2) >= old_element_id + 3
        @assert element_id >= 1
        @assert size(u, 1) == nvariables(equations)
        @assert size(u, 2) >= element_id
    end

    # Project from lower left element
    acc = get_node_vars(old_u, equations, solver, lower_left_id)

    # Project from lower right element
    acc += get_node_vars(old_u, equations, solver, lower_right_id)

    # Project from upper left element
    acc += get_node_vars(old_u, equations, solver, upper_left_id)

    # Project from upper right element
    acc += get_node_vars(old_u, equations, solver, upper_right_id)

    # Update value
    set_node_vars!(u, 1 / 4 * acc, equations, solver, element_id)
end
end # @muladd
