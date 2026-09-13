###############################################################################
# Stationary autocorrelation curves from the Galerkin transfer operator
#
# For an observable f, the invariant mean and centred observable are
#
#     m_ε = ∫ f ρ_ε,        f₀ = f - m_ε,
#
# and the stationary autocorrelation at lag n is, by Perron-Frobenius/Koopman
# duality,
#
#     C_ε(n) = ∫ f₀(Tⁿx) f₀(x) dμ_ε(x)
#            = ∫ f₀(x) · P_ε^n(f₀ ρ_ε)(x) dx.
#
# This file evaluates the right-hand side in the finite-dimensional Galerkin
# space.  Let
#
#     ρ_N = Σ_j c_j φ_j
#
# be the Galerkin approximation of the invariant density, and set
#
#     h₀ = f₀ ρ_N.
#
# The computation has three steps:
#
# 1. Project h₀ onto the B-spline space.
#
#    Write
#
#        Π_N h₀ = Σ_j a₀[j] φ_j.
#
#    The L² projection equations are
#
#        M a₀ = b,
#
#    where
#
#        M[i,j] = ⟨φ_i, φ_j⟩,
#        b[i]   = ⟨φ_i, h₀⟩ = ∫ φ_i f₀ ρ_N.
#
# 2. Propagate the coefficient vector of the projected function with the discrete transfer operator.
#
#    With the matrix convention
#
#        G[j,i] = ⟨P_ε φ_i, φ_j⟩,
#
#    one Galerkin application of P_ε sends
#
#        a_n ↦ a_{n+1} = M \ (G * a_n).
#
#    Hence, with A = M \ G, we have
#
#        a_n = A^n a₀.
#
# 3. Extract the autocorrelation from the propagated coefficients.
#
#    If
#
#        h_n = Σ_j a_n[j] φ_j,
#
#    then
#
#        C_N(n) = ∫ f₀ h_n
#               = Σ_j a_n[j] ∫ f₀ φ_j
#               = ℓᵀ a_n,
#
#    where
#
#        ℓ[j] = ⟨f₀, φ_j⟩ = ∫ f₀ φ_j.
#
#    Thus ℓ represents the final linear functional h ↦ ∫ f₀ h in coefficient
#    space: b is used to put the initial function into the spline space, while
#    ℓ is used to pull the desired scalar correlation back out.
#
# Altogether,
#
#     C_N(n) = ℓᵀ A^n a₀
#            = ℓᵀ (M \ G)^n (M \ b).
#
# M is Cholesky-factored once.  The factorization is used to compute a₀ and to
# form A = M \ G up front, so each lag then requires only a single matrix-vector
# product.  M⁻¹ is never formed explicitly.
###############################################################################


using LinearAlgebra: cholesky, Symmetric, dot, mul!


"""
    CorrelationCurve{T<:AbstractFloat}

Output of `correlation_curve`.

For the centred observable `f₀`, the Galerkin correlation calculation is

    h₀ = f₀ ρ_N
    M a₀ = b,                    b[i] = ∫ φ_i h₀
    aₙ₊₁ = M \\ (G aₙ)
    C_N(n) = ℓᵀ aₙ,              ℓ[j] = ∫ f₀ φ_j.

Thus `aₙ` is the coefficient vector of the propagated Galerkin approximation
to `P_ε^n h₀`, and `C[n+1]` is obtained by applying the correlation functional
`ℓᵀ` to that coefficient vector.

Fields
------
- `lags`         : `0:max_lag`

- `C`            : `C[n+1] = C_N(n) = ℓᵀ aₙ`

- `mass`         : `mass[n+1] = ∫ hₙ dx`, where

                       hₙ = Σ_j aₙ[j] φ_j.

                   Since the B-splines form a partition of unity,
                   `1 ∈ span{φ_j}`, so `mass[1]` equals `∫ f₀ ρ_N` exactly
                   (not merely to projection accuracy). It is therefore the
                   centering residual.

                   The discrete operator preserves this mass because

                       Σ_j G[j,i] = ∫ P_ε φ_i = ∫ φ_i.

                   Hence the whole `mass` vector should remain close to zero.
                   Its residual primarily measures mass-conservation error in
                   the assembled transfer matrix `G`, rather than floating-point
                   round-off alone.

- `mean`         : `m_N = ∫ f ρ_N`

- `variance`     : `∫ f₀² ρ_N`, computed directly by quadrature.

                   At lag zero,

                       C[1] = ⟨f₀, Π_N(f₀ ρ_N)⟩,

                   which differs from the directly computed variance by

                       ⟨f₀ - Π_N f₀, f₀ρ_N - Π_N(f₀ρ_N)⟩.

                   This is the product of two L² projection errors. For a smooth
                   observable `f`, the difference is therefore typically much
                   smaller than either individual projection error.

- `density_mass` : `∫ ρ_N dx`, which should be 1 for a normalised density.
"""
struct CorrelationCurve{T<:AbstractFloat}
    lags         :: UnitRange{Int}
    C            :: Vector{T}
    mass         :: Vector{T}
    mean         :: T
    variance     :: T
    density_mass :: T
end


Base.length(cc::CorrelationCurve) = length(cc.C)


function Base.show(io::IO, cc::CorrelationCurve{T}) where {T}
    print(io, "CorrelationCurve{", T, "}(lags ", cc.lags,
          ", C(0) = ", cc.C[1],
          ", var = ", cc.variance,
          ", max|∫h| = ", maximum(abs, cc.mass), ")")
end


"""
    _basis_inner_products(splines, x, wg) -> Vector{T}

Compute the vector of inner products of a function `g` with the B-spline basis,

    v[j] = ⟨φ_j, g⟩ = ∫ φ_j(x) g(x) dx,

by quadrature.

If `x[q]` and `w[q]` are the quadrature nodes and weights, then `wg[q]`
contains

    w[q] * g(x[q]),

so

    v[j] ≈ Σ_q wg[q] * φ_j(x[q]).

This helper is used to construct several coefficient-space functionals and
projection right-hand sides. In particular, the correlation calculation uses

    b[i] = ∫ φ_i f₀ ρ_N

for the projection of `h₀ = f₀ρ_N`, and

    ℓ[j] = ∫ φ_j f₀

for the final correlation functional.

`x` must be sorted, as it is when produced by `density_quadrature`. Each
B-spline is exactly zero outside `ϕ.support`, so restricting the quadrature
sum to nodes inside the support is exact and reduces the cost from `O(n·Q)`
to `O((p+1)·Q)`.
"""
function _basis_inner_products(
    splines :: Vector{<:SingleBSpline{T}},
    x       :: Vector{T},
    wg      :: Vector{T},
) ::Vector{T} where {T<:AbstractFloat}

    length(x) == length(wg) ||
        throw(DimensionMismatch("x and wg must have the same length"))

    n = length(splines)
    v = zeros(T, n)

    @threads for j in 1:n
        ϕ   = splines[j]
        acc = zero(T)

        @inbounds for (sa, sb) in ϕ.support
            lo = searchsortedfirst(x, sa)
            hi = searchsortedlast(x, sb)

            for q in lo:hi
                acc += wg[q] * ϕ(x[q])
            end
        end

        v[j] = acc
    end

    return v
end


"""
    _correlation_setup(f, splines, c, ε; n_quad, atol, refine)

Centre `f` against the Galerkin invariant density `ρ_N = Σ_j c_j φ_j` and
assemble the vectors needed by the correlation calculation:

    b[i]    = ∫ φ_i f₀ ρ_N      projection right-hand side, so M a₀ = b,
    ℓ[j]    = ∫ f₀ φ_j          correlation functional, so C_N(n) = ℓᵀ aₙ,
    mvec[j] = ∫ φ_j             mass functional, so ∫ hₙ = mvecᵀ aₙ.

Also computes

    mean         = ∫ f ρ_N,
    variance     = ∫ f₀² ρ_N,
    density_mass = ∫ ρ_N.

Returns `(ℓ, b, mvec, mean, variance, density_mass)`.
"""
function _correlation_setup(
    f,
    splines :: Vector{<:SingleBSpline{T}},
    c       :: AbstractVector{T},
    ε       :: T;
    n_quad  :: Int = 32,
    atol    :: T   = eps(T),
    refine  :: Union{Nothing,EndpointRefinement{T}} = nothing,
) where {T<:AbstractFloat}

    length(c) == length(splines) ||
        throw(DimensionMismatch(
            "c has $(length(c)) entries but the basis has $(length(splines))"
        ))

    # Build one quadrature mesh and one set of quadrature nodes.
    # q0.w contains the bare quadrature weights and is used for integrals with
    # respect to dx. The density-weighted rule dq folds ρ_N into those weights
    # and is used for integrals with respect to ρ_N(x) dx.
    mesh = build_integration_mesh(
        splines[1].knots,
        ε;
        atol=atol,
        refine=refine,
    )

    q0 = density_quadrature(
        x -> one(T),
        mesh;
        n_quad=n_quad,
        atol=atol,
    )

    # Galerkin invariant density: ρ_N = Σ_j c_j φ_j.
    pdf = BSK.Spline(
        splines[1].basis,
        collect(T, c),
    )

    dq = DensityQuadrature{T}(
        q0.mesh,
        q0.x,
        q0.w .* pdf.(q0.x),
    )

    density_mass = total_mass(dq)

    # Centre the observable:

    fvals  = map(x -> T(f(x)), q0.x)
    mean   = dot(dq.w, fvals)
    f0vals = fvals .- mean

    # Direct quadrature of Var(f) = ∫ f₀² ρ_N.
    variance = zero(T)

    @inbounds for q in eachindex(f0vals)
        variance += dq.w[q] * f0vals[q]^2
    end

    # Output functional:

    ℓ = _basis_inner_products(
        splines,
        q0.x,
        q0.w .* f0vals,
    )

    # Initial projection right-hand side:

    b = _basis_inner_products(
        splines,
        q0.x,
        dq.w .* f0vals,
    )

    # Mass functional in coefficient space:

    mvec = T[ϕ.mass for ϕ in splines]

    return ℓ, b, mvec, mean, variance, density_mass
end


"""
    correlation_curve(f, splines, M, G, c, ε, max_lag;
                      n_quad = 32, atol = eps(T), refine = nothing) -> CorrelationCurve{T}

Compute the Galerkin approximation of the stationary autocorrelation curve

    C_N(n) = ℓᵀ aₙ,

with

    M a₀ = b,
    aₙ₊₁ = (M \\ G) aₙ.

The setup routine constructs `b`, `ℓ`, and the diagnostics from the Galerkin
invariant density `ρ_N = Σ_j c_j φ_j`.

`M` is Cholesky-factored once and used to compute both `a₀ = M \\ b` and the
coefficient-space transfer operator `A = M \\ G`, so each subsequent lag
requires one matrix-vector product.

Arguments
---------
- `f`       : observable.
- `splines` : B-spline basis functions.
- `M`       : mass matrix, `M[i,j] = ⟨φ_i, φ_j⟩`.
- `G`       : transfer matrix, `G[j,i] = ⟨P_ε φ_i, φ_j⟩`.
- `c`       : coefficients of the normalised Galerkin invariant density.
- `ε`       : noise half-width used to construct the quadrature mesh.
- `max_lag` : largest lag to evaluate.
- `n_quad`  : Gauss-Legendre nodes per mesh panel.
- `refine`  : optional endpoint refinement of the quadrature mesh.

See `CorrelationCurve` for the returned diagnostics.
"""
function correlation_curve(
    f,
    splines :: Vector{<:SingleBSpline{T}},
    M       :: AbstractMatrix{T},
    G       :: AbstractMatrix{T},
    c       :: AbstractVector{T},
    ε       :: T,
    max_lag :: Int;
    n_quad  :: Int = 32,
    atol    :: T   = eps(T),
    refine  :: Union{Nothing,EndpointRefinement{T}} = nothing,
) ::CorrelationCurve{T} where {T<:AbstractFloat}

    max_lag >= 0 ||
        throw(ArgumentError("max_lag must be ≥ 0, got $max_lag"))

    n = length(splines)

    size(M) == (n, n) ||
        throw(DimensionMismatch("M must be $n×$n, got $(size(M))"))

    size(G) == (n, n) ||
        throw(DimensionMismatch("G must be $n×$n, got $(size(G))"))

    # Build the projection right-hand side b, correlation functional ℓ,
    # mass functional, and diagnostics.
    ℓ, b, mvec, mean, variance, density_mass =
        _correlation_setup(
            f,
            splines,
            c,
            ε;
            n_quad=n_quad,
            atol=atol,
            refine=refine,
        )

    # Initial projection a₀ = M \ b and coefficient-space operator A = M \ G.
    F = cholesky(Symmetric(M))
    a = F \ b
    A = F \ G

    C    = Vector{T}(undef, max_lag + 1)
    mass = Vector{T}(undef, max_lag + 1)
    tmp  = similar(a)

    for k in 0:max_lag
        C[k+1]    = dot(ℓ, a)
        mass[k+1] = dot(mvec, a)

        k == max_lag && break

        # Propagate aₖ → aₖ₊₁.
        mul!(tmp, A, a)
        a, tmp = tmp, a
    end

    return CorrelationCurve{T}(
        0:max_lag,
        C,
        mass,
        mean,
        variance,
        density_mass,
    )
end

"""
    _default_fit_lags(cc::CorrelationCurve) -> Vector{Int}

The last third of the lags, the default window for `estimate_decay_rate`.
"""
function _default_fit_lags(cc::CorrelationCurve) ::Vector{Int}
    n = length(cc.lags)
    return collect(cc.lags[max(1, n - cld(n, 3) + 1):n])
end


"""
    estimate_decay_rate(cc::CorrelationCurve{T}; lags = <last third>) -> (rate, amplitude)

Least-squares fit of `log|C_N(n)|` against `n` over `lags`, returning
`rate = exp(slope)` and `amplitude = exp(intercept)`, so that
`C_N(n) ≈ ± amplitude · rateⁿ` on the fitted window.

Lags whose `|C|` is zero or non-finite in the log are dropped from the fit.
"""
function estimate_decay_rate(
    cc   :: CorrelationCurve{T};
    lags :: AbstractVector{Int} = _default_fit_lags(cc),
) where {T<:AbstractFloat}

    ns = T[]
    ys = T[]
    for k in lags
        k in cc.lags || throw(ArgumentError("lag $k is outside $(cc.lags)"))
        y = log(abs(cc.C[k+1]))
        isfinite(y) || continue
        push!(ns, T(k))
        push!(ys, y)
    end

    length(ns) >= 2 ||
        throw(ArgumentError("need at least 2 usable lags for the fit, got $(length(ns))"))

    X = hcat(ones(T, length(ns)), ns)
    β = X \ ys

    return (rate = exp(β[2]), amplitude = exp(β[1]))
end
