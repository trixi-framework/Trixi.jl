# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
    @doc raw"""
        PolytropicEulerEquations2D(gamma, kappa)

    The polytropic Euler equations
    ```math
    \partial t
    \begin{pmatrix}
    \rho \\ \rho v_1 \\ \rho v_2
    \end{pmatrix}
    +
    \partial x
    \begin{pmatrix}
     \rho v_1 \\ \rho v_1^2 + \kappa\rho^\gamma \\ \rho v_1 v_2
    \end{pmatrix}
    +
    \partial y
    \begin{pmatrix}
    \rho v_2 \\ \rho v_1 v_2 \\ \rho v_2^2 + \kappa\rho^\gamma
    \end{pmatrix}
    =
    \begin{pmatrix}
    0 \\ 0 \\ 0
    \end{pmatrix}
    ```
    for an ideal gas with ratio of specific heats `gamma`
    in two space dimensions.
    Here, ``\rho`` is the density and ``v_1`` and`v_2` the velocities and
    ```math
    p = \kappa\rho^\gamma
    ```
    the pressure, which we replaced using this relation.

    """
    struct PolytropicEulerEquations2D{RealT <: Real, RealT <: Real} <:
           AbstractPolytropicEulerEquations{2, 3}
        gamma::RealT               # ratio of specific heats
        kappa::RealT               # fluid scaling factor

        function PolytropicEulerEquations2D(gamma, kappa)
            new{typeof(gamma), typeof(kappa)}(gamma, kappa)
        end
    end

    function varnames(::typeof(cons2cons), ::PolytropicEulerEquations2D)
        ("rho", "rho_v1", "rho_v2")
    end
    varnames(::typeof(cons2prim), ::PolytropicEulerEquations2D) = ("rho", "v1", "v2")

    # Calculate 1D flux for a single point in the normal direction
    # Note, this directional vector is not normalized
    @inline function flux(u, normal_direction::AbstractVector,
                          equations::PolytropicEulerEquations2D)
        rho_e = last(u)
        rho, v1, v2 = cons2prim(u, equations)
        p = pressure(u, equations)

        v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
        rho_v_normal = rho * v_normal
        f1 = rho_v_normal
        f2 = rho_v_normal * v1 + p * normal_direction[1]
        f3 = rho_v_normal * v2 + p * normal_direction[2]
        return SVector(f1, f2, f3)
    end

    @inline function flux_ranocha(u_ll, u_rr, normal_direction::AbstractVector,
                                  equations::PolytropicEulerEquations2D)
        # Unpack left and right state
        rho_ll, v1_ll, v2_ll = cons2prim(u_ll, equations)
        rho_rr, v1_rr, v2_rr = cons2prim(u_rr, equations)
        p_ll = equations.kappa * rho_ll^equations.gamma
        p_rr = equations.kappa * rho_rr^equations.gamma
        v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
        v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

        # Compute the necessary mean values
        rho_mean = ln_mean(rho_ll, rho_rr)
        # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
        # in exact arithmetic since
        #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
        #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
        inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
        v1_avg = 0.5 * (v1_ll + v1_rr)
        v2_avg = 0.5 * (v2_ll + v2_rr)
        p_avg = 0.5 * (p_ll + p_rr)

        # Calculate fluxes depending on normal_direction
        f1 = rho_mean * 0.5 * (v_dot_n_ll + v_dot_n_rr)
        f2 = f1 * v1_avg + p_avg * normal_direction[1]
        f3 = f1 * v2_avg + p_avg * normal_direction[2]

        return SVector(f1, f2, f3)
    end

    @inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                         equations::PolytropicEulerEquations2D)
        rho_ll, v1_ll, v2_ll = cons2prim(u_ll, equations)
        rho_rr, v1_rr, v2_rr = cons2prim(u_rr, equations)
        p_ll = equations.kappa * rho_ll^equations.gamma
        p_rr = equations.kappa * rho_rr^equations.gamma

        v_normal_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
        v_normal_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

        norm_ = norm(normal_direction)
        # The v_normals are already scaled by the norm
        λ_min = v_normal_ll - sqrt(equations.gamma * p_ll / rho_ll) * norm_
        λ_max = v_normal_rr + sqrt(equations.gamma * p_rr / rho_rr) * norm_

        return λ_min, λ_max
    end

    @inline function max_abs_speeds(u, equations::PolytropicEulerEquations2D)
        rho, v1, v2 = cons2prim(u, equations)
        c = sqrt(equations.gamma * equations.kappa * rho^(equations.gamma - 1))

        return abs(v1) + c, abs(v2) + c
    end

    # Convert conservative variables to primitive
    @inline function cons2prim(u, equations::PolytropicEulerEquations2D)
        rho, rho_v1, rho_v2 = u

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho

        return SVector(rho, v1, v2)
    end

    # Convert conservative variables to entropy
    @inline function cons2entropy(u, equations::PolytropicEulerEquations2D)
        rho, rho_v1, rho_v2 = u

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v_square = v1^2 + v2^2
        p = pressure(u, equations)
        s = rho / 2 * v_square +
            rho * equations.kappa * rho^(equations.gamma - 1) / (equations.gamma - 1)
        rho_p = rho / p

        w1 = (equations.gamma - s) * (equations.gamma - 1) - 0.5 * rho_p * v_square
        w2 = rho_p * v1
        w3 = rho_p * v2

        return SVector(w1, w2, w3)
    end

    # Convert primitive to conservative variables
    @inline function prim2cons(prim, equations::PolytropicEulerEquations2D)
        rho, v1, v2 = prim
        rho_v1 = rho * v1
        rho_v2 = rho * v2
        return SVector(rho, rho_v1, rho_v2)
    end

    @inline function density(u, equations::PolytropicEulerEquations2D)
        rho = u[1]
        return rho
    end

    @inline function pressure(u, equations::PolytropicEulerEquations2D)
        rho, rho_v1, rho_v2 = u
        p = equations.kappa * rho^equations.gamma
        return p
    end
end # @muladd
