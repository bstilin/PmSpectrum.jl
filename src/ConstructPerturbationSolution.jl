import SpecialFunctions: loggamma


"""
    exp_times_upper_gamma(s::T, z::T;
        tol::T = 64eps(T),
        maxiter::Integer = 512,
        asymp_cutoff::T = T(128),
    )::T where {T<:AbstractFloat}

Compute the scaled upper incomplete gamma function

    E(s,z) = exp(z) * Γ(s,z)

natively in the input floating-point type `T` (works for `Float64`, `Double64`,
and `BigFloat` — `loggamma` has accurate methods for each).

This implementation avoids `gamma_inc` (which is Float64-only) and is intended
for the parameter range

    2/3 < s < 3/2,     z ≥ 0,

as used by `construct_perturbation_solution` (s = 2/β and 3/β, β = α+2).

It uses three representations:

1. For `z < s + 1`, a convergent lower-tail series,

       E(s,z) = exp(z)Γ(s) - z^s * Σ z^n / (s(s+1)...(s+n)).

2. For `z ≥ s + 1`, a scaled upper-tail continued fraction,

       E(s,z) = z^s H(s,z),

   evaluated by modified Lentz iteration.

3. For sufficiently large `z`, a large-`z` asymptotic expansion is tried first.
   If it does not reach the requested tolerance safely, the method falls back to
   the continued fraction.

All arithmetic is performed in the input type `T`, and the returned value has
type `T`.
"""
function exp_times_upper_gamma(
    s::T,
    z::T;
    tol::T = T(64) * eps(T),
    maxiter::Integer = 512,
    asymp_cutoff::T = T(128),
)::T where {T<:AbstractFloat}

    zeroT = zero(T)
    oneT = one(T)

    s > zeroT || throw(ArgumentError("Require s > 0"))
    z >= zeroT || throw(ArgumentError("Require z >= 0"))

    if iszero(z)
        return exp(loggamma(s))
    end

    if z < s + oneT
        return _E_lower_series(s, z; tol=tol, maxiter=maxiter)
    end

    if z >= asymp_cutoff
        val, ok = _E_large_z_asymptotic(s, z; tol=tol, maxiter=maxiter)
        ok && return val
    end

    return _E_continued_fraction(s, z; tol=tol, maxiter=maxiter)
end


function _E_lower_series(
    s::T,
    z::T;
    tol::T,
    maxiter::Integer,
)::T where {T<:AbstractFloat}

    oneT = one(T)

    term = inv(s)
    sum_terms = term

    @inbounds for n in 1:maxiter
        term *= z / (s + T(n))
        new_sum = sum_terms + term

        if abs(term) <= tol * max(oneT, abs(new_sum))
            sum_terms = new_sum
            break
        end

        sum_terms = new_sum

        if n == maxiter
            throw(ErrorException("lower incomplete-gamma series failed to converge"))
        end
    end

    return exp(z + loggamma(s)) - z^s * sum_terms
end


function _E_continued_fraction(
    s::T,
    z::T;
    tol::T,
    maxiter::Integer,
)::T where {T<:AbstractFloat}

    oneT = one(T)

    tiny = sqrt(floatmin(T))

    # H(s,z) = 1 / (b₀ + a₁/(b₁ + a₂/(b₂ + ...))).
    #
    # b₀ = z + 1 - s,
    # aₙ = n(s - n),
    # bₙ = z + 2n + 1 - s.
    b0 = z + oneT - s
    abs(b0) > tiny || (b0 = tiny)

    h = inv(b0)
    c = inv(tiny)
    d = h

    @inbounds for n in 1:maxiter
        nT = T(n)

        a_n = nT * (s - nT)
        b_n = z + T(2n + 1) - s

        d = b_n + a_n * d
        abs(d) > tiny || (d = tiny)

        c = b_n + a_n / c
        abs(c) > tiny || (c = tiny)

        d = inv(d)
        delta = c * d
        h *= delta

        if abs(delta - oneT) <= tol
            return z^s * h
        end
    end

    throw(ErrorException("upper incomplete-gamma continued fraction failed to converge"))
end


function _E_large_z_asymptotic(
    s::T,
    z::T;
    tol::T,
    maxiter::Integer,
)::Tuple{T,Bool} where {T<:AbstractFloat}

    oneT = one(T)

    # E(s,z) ~ z^(s-1) * Σ b_k,
    # b₀ = 1,
    # b_{k+1} = b_k * (s - 1 - k) / z.
    term = oneT
    poly = oneT
    prev_abs = abs(term)

    @inbounds for k in 1:maxiter
        term_next = term * (s - T(k)) / z
        next_abs = abs(term_next)

        # Since this is an asymptotic expansion, do not keep adding terms
        # after they stop decreasing.
        if next_abs > prev_abs
            return z^(s - oneT) * poly, false
        end

        new_poly = poly + term_next

        if next_abs <= tol * max(oneT, abs(new_poly))
            return z^(s - oneT) * new_poly, true
        end

        term = term_next
        poly = new_poly
        prev_abs = next_abs
    end

    return z^(s - oneT) * poly, false
end


"""
    construct_perturbation_solution(α::T, ϵ::T, rho_half::T; order=:second) -> y::Function
    construct_perturbation_solution(α::T, rho_half::T)                      -> y::Function
where {T<:AbstractFloat}

Return a closure `y(x::T)::T` representing a perturbation solution for the
Symmetric PM Map invariant density near the neutral fixed point with exponent 
`α` (require 0 < α < 1), calibrated by the constraint `f(1/2) = rho_half`.

# Arguments
- `α::T`          : exponent, must satisfy 0 < α < 1.
- `epsilon::T`    : noise half-width of the uniform distribution, i.e. the noise is uniformly distributed
                    on [-ε, ε]. Must satisfy 1/2 ≥ epsilon ≥ 0; pass 0 or no epsilon
                    for argument for the noisless solution.
- `order::Symbol` : perturbation order; `:first` (ν = 2/β term only) or `:second` (default,
                    includes the ν = 3/β correction).

# Returns
- `y::Function` such that `y(x::T)::T` evaluates the perturbation solution.

"""
function construct_perturbation_solution end

function construct_perturbation_solution(α::T, epsilon::T, rho_half::T; order::Symbol=:second) where {T<:AbstractFloat}

    (zero(T) < α < one(T)) || throw(ArgumentError("Require 0 < α < 1"))
    T(.5) ≥ epsilon ≥ zero(T)            || throw(ArgumentError("Require 1/2 ≥ ε ≥ 0"))

    epsilon == zero(T) && return construct_perturbation_solution(α, rho_half; order=order)

    oneT   = one(T)
    twoT   = T(2)
    threeT = T(3)

    β = α + twoT
    a = exp2(α)
    L_ε = Utils.boundary_layer_scale(α, epsilon)

    b = rho_half / (α + twoT)
    c = rho_half * α * (oneT + α) / (α + twoT)^3

    β_over_a = β / a
    K2 = (b/β)   * (β_over_a)^( twoT/β)
    K3 = (c/β) * (β_over_a)^(threeT/β)

    @inline function U(t::T)::T
        z = (a/β) * t^β

        if order == :first
            return K2 * exp_times_upper_gamma(twoT/β,   z)
        elseif order == :second
            return K2 * exp_times_upper_gamma(twoT/β,   z) +
                L_ε * K3 * exp_times_upper_gamma(threeT/β, z)
        else
            throw(ArgumentError("Invalid order: expected :first or :second, got $order"))
        end
    end

    @inline function y(x::T)::T
        t = x / L_ε
        return U(t) * L_ε^(-α)
    end

    return y
end

function construct_perturbation_solution(α::T, rho_half::T; order::Symbol=:second) where {T<:AbstractFloat}

    oneT = one(T)
    twoT = T(2)
    threeT = T(3)

    K2 = rho_half * inv( twoT^α * (twoT+α) )

    if order == :first
        return x -> K2 * x^(-α)
    elseif order == :second
        K3 = rho_half * α * (oneT + α) * inv(twoT^α * (twoT + α)^threeT)
        return x -> K2 * x^(-α) + K3 * x^(oneT-α)
    elseif order == :oseen
        return x -> K2 * x^(-α) + rho_half / (twoT + α) * ( oneT + α / twoT)
    else
        throw(ArgumentError("Invalid order: expected :first or :second, got $order"))
    end
end


###############################################################################
# Global density approximation ρ̂_ε
###############################################################################

"""
    _smooth_hermite(t::T) -> T

Smooth Hermite blending polynomial W(t) = 6t⁵ − 15t⁴ + 10t³ on [0,1].
Satisfies W(0) = 0, W(1) = 1, W'(0) = W'(1) = W''(0) = W''(1) = 0.
"""
@inline _smooth_hermite(t::T) where {T<:AbstractFloat} =
    t^3 * (T(10) - t * (T(15) - T(6) * t))


"""
    construct_global_density_hat(α, ε, ref_density; c_match=10, rho_half, order) -> Function

Return a closure `x::T -> T` implementing the five-region global density
approximation ρ̂_ε that blends the closed-form boundary-layer solution U_ε with
a numerically computed reference density ρ̃_ε.

`ref_density` is any callable (e.g. `x -> S_ref(x)` for a BSplineKit Spline `S_ref`).
The boundary-layer scale L_ε is computed internally.

Piecewise definition (c = c_match > 1, W = _smooth_hermite):

| Region       | ρ̂_ε(x)                                               |
|------------- |-------------------------------------------------------|
| [0, L_ε]     | U_ε(x)                                                |
| [L_ε, cL_ε]  | (1-W(t₂)) U_ε(x)    + W(t₂) ref(x),  t₂=(x-L_ε)/((c-1)L_ε) |
| [cL_ε, 1-cL_ε] | ref(x)                                                |
| [1-cL_ε, 1-L_ε] | (1-W(t₄)) U_ε(1-x)  + W(t₄) ref(x),  t₄=(1-L_ε-x)/((c-1)L_ε) |
| [1-L_ε, 1]   | U_ε(1-x)                                              |

In region 4, t₄ decreases from 1 to 0 as x goes from 1-cL to 1-L, so W(1)=1
gives ref(x) at the left edge and W(0)=0 gives U_ε(1-x) at the right edge.

Arguments
---------
- `α`, `ε`       : PM exponent and noise half-width (must satisfy 0 < α < 1, ε > 0).
- `ref_density`  : callable ρ̃_ε(x::T)::T (e.g. a wrapped BSplineKit Spline).
- `c_match`      : matching constant c > 1 (default 10).
- `rho_half`     : value of the density at x=1/2; defaults to `ref_density(1/2)`.
- `order`        : perturbation order passed to `construct_perturbation_solution`.
"""
function construct_global_density_hat(
    α          :: T,
    ε          :: T,
    ref_density;
    c_match    :: Real              = 10,
    rho_half   :: Union{Real,Nothing} = nothing,
    order      :: Symbol = :second,
) where {T<:AbstractFloat}

    c_match_T  = T(c_match)
    c_match_T > one(T) || throw(ArgumentError("c_match must be > 1, got $c_match"))
    L_ε        = boundary_layer_scale(α, ε)
    rho_half_T = rho_half === nothing ? T(ref_density(one(T) / T(2))) : T(rho_half)

    cL_ε    = c_match_T * L_ε
    U_eps   = construct_perturbation_solution(α, ε, rho_half_T; order=order)
    inv_cL_ε  = inv((c_match_T - one(T)) * L_ε)   # 1 / ((c-1)L_ε), precomputed

    function rho_hat(x::T)::T
        if x ≤ L_ε
            return U_eps(x)
        elseif x ≤ cL_ε
            t = (x - L_ε) * inv_cL_ε
            return (one(T) - _smooth_hermite(t)) * U_eps(x) + _smooth_hermite(t) * ref_density(x)
        elseif x ≤ one(T) - cL_ε
            return ref_density(x)
        elseif x ≤ one(T) - L_ε
            t = (one(T) - L_ε - x) * inv_cL_ε   # decreases from 1 to 0 across this region
            return (one(T) - _smooth_hermite(t)) * U_eps(one(T) - x) + _smooth_hermite(t) * ref_density(x)
        else
            return U_eps(one(T) - x)
        end
    end

    return rho_hat
end