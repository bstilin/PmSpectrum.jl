# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

module ResolventNorm

using LinearAlgebra
using GenericLinearAlgebra   # extends svdvals / cond to Double64 and BigFloat
using ..Bases: SingleBSpline

export orthonormal_complement, l2_orthonormal_form, reduced_resolvent_norm


"""
    mass_tolerance(T) -> T

Largest mass defect `‖q* B‖₂ / σ_max(B₀)` still attributable to assembly and
round-off, above which `reduced_resolvent_norm` warns.

These are set from measurement, not from `eps(T)`. The assembled `G` satisfies
its mass-conservation identity only to about `2e-17` in `Double64` — a floor in
the assembly path, not Gauss-Legendre truncation, and insensitive to `n_quad` —
so an `eps(Double64)`-sized threshold would fire on every run. Observed defects
are ~1e-17 (Double64) and ~1e-14 to 1e-12 (Float64).
"""
mass_tolerance(::Type{Float64}) = 1e-10
mass_tolerance(::Type{T}) where {T<:AbstractFloat} = T(1e-14)


"""
    orthonormal_complement(q) -> Q₀

Euclidean orthonormal basis for `q⊥`, as an `n × (n-1)` matrix satisfying
`Q₀* Q₀ = I` and `Q₀* q = 0`.

Obtained from the Householder QR factorisation of `q` viewed as an `n × 1`
matrix: the first column of the full orthogonal factor is `±q/‖q‖`, so the
remaining `n-1` columns span its orthogonal complement.

Requires `length(q) ≥ 2` and `q ≠ 0`. The zero vector is rejected rather than
handled: its orthogonal complement is all of `Rⁿ`, so no `n × (n-1)` matrix
represents it and the QR would hand back an arbitrary one without complaint.
"""
function orthonormal_complement(q::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(q)
    n ≥ 2 || throw(ArgumentError("Require length(q) ≥ 2, got $n"))
    iszero(q) && throw(ArgumentError(
        "q must be nonzero: the orthogonal complement of the zero vector is all " *
        "of Rⁿ, not an (n-1)-dimensional subspace"))

    Q = qr(reshape(q, n, 1)).Q * Matrix{T}(I, n, n)
    return Q[:, 2:n]
end


"""
    l2_orthonormal_form(M, G, m) -> NamedTuple

Move the Galerkin pair `(M, G)` and the integration functional `m` into
Euclidean L²-orthonormal coordinates `y = L* D⁻¹ c`.

Equilibrates with `D = diag(M)^(-1/2)`, Cholesky-factors `M_s = D M D = L L*`,
and returns

    B = L⁻¹ (M_s - G_s) L⁻*     representing  I - P_{ε,N},
    q = L⁻¹ D m,  normalised    representing  the mass functional,

together with `κ(M)` and `κ(M_s)` so the benefit of the equilibration is
visible. Both inverses are applied as triangular solves.

Arguments
- `M` : n×n symmetric positive-definite mass matrix, `M[i,j] = ⟨φ_i, φ_j⟩`
- `G` : n×n transfer matrix, `G[j,i] = ⟨P_ε φ_i, φ_j⟩`
- `m` : length-n integration functional, `m[i] = ∫₀¹ φ_i dx`

`M`'s definiteness is checked in two stages: its diagonal must be strictly
positive *before* `D` is formed, since `sqrt` of a nonpositive entry would poison
everything downstream, and the Cholesky then catches the rest.

Returns `(B, q, cond_mass, cond_mass_scaled)`.
"""
function l2_orthonormal_form(
    M :: AbstractMatrix{T},
    G :: AbstractMatrix{T},
    m :: AbstractVector{T},
) where {T<:AbstractFloat}

    n = size(M, 1)
    size(M, 2) == n   || throw(ArgumentError("M must be square, got $(size(M))"))
    size(G) == size(M) ||
        throw(ArgumentError("G must have the same size as M, got $(size(G)) and $(size(M))"))
    length(m) == n ||
        throw(ArgumentError("m must have length $n, got $(length(m))"))
    issymmetric(M) || throw(ArgumentError("M must be symmetric"))

    # Checked here rather than left to the Cholesky below: a nonpositive diagonal
    # entry makes `sqrt.(diag(M))` produce NaN/Inf first, and `cholesky` on a
    # NaN-laden matrix does not reliably raise the PosDefException that the catch
    # translates — so the clean error would be bypassed.
    d = diag(M)
    all(>(zero(T)), d) || throw(ArgumentError(
        "M must be positive definite: diagonal entry $(argmin(d)) is $(minimum(d))"))

    # D: each basis function replaced by its individually L²-normalised version.
    D = Diagonal(inv.(sqrt.(d)))

    Ms = Symmetric(D * M * D)
    Ks = D * (M - G) * D
    ms = D * m

    cond_mass        = cond(Matrix(M))
    cond_mass_scaled = cond(Matrix(Ms))

    L = try
        cholesky(Ms).L
    catch e
        e isa PosDefException && throw(ArgumentError("M must be positive definite"))
        rethrow()
    end

    B = (L \ Ks) / adjoint(L)    # L⁻¹ K_s L⁻*
    q = normalize(L \ ms)        # L⁻¹ m_s

    return (B = B, q = q, cond_mass = cond_mass, cond_mass_scaled = cond_mass_scaled)
end


"""
    reduced_resolvent_norm(M, G, m)       -> NamedTuple
    reduced_resolvent_norm(M, G, splines) -> NamedTuple
    reduced_resolvent_norm(result)        -> NamedTuple

The L²-operator norm of the reduced resolvent of the noisy Galerkin transfer
operator at z = 1,

    M_{ε,N} = ‖ [ (I - P_{ε,N})|_{V_{N,0}} ]⁻¹ ‖_{L² → L²} = 1 / σ_min(B₀),

where `B₀ = Q₀* B Q₀` is `I - P_{ε,N}` restricted to the zero-mass space, in the
L²-orthonormal coordinates built by `l2_orthonormal_form`.

`m` is the exact integration functional `m[i] = ∫₀¹ φ_i dx`. Passing a vector of
`SingleBSpline` takes it from the precomputed `ϕ.mass` fields; passing an
`InvariantDensityResult` rehydrates the basis and does the same.

Diagnostics returned alongside the norm
- `mass_error` : `‖q* B‖₂ / σ_max(B₀)`, dimensionless. Mass preservation means
  `q* B = 0` exactly, so this should sit at the assembly floor. It is reported,
  never projected away; a defect above `mass_tolerance(T)` raises a warning, and
  in that case the restricted SVD is not measuring what it claims to.

  The denominator is the largest singular value of the *restricted* `B₀`, not of
  `B`. These differ: `q* B = 0` says `range(B) ⊆ q⊥`, which says nothing about
  `Bq`, and in the basis `[q Q₀]` the matrix `B` is unitarily equivalent to
  `[0 0; Q₀*Bq B₀]`, so `σ_max(B) ≥ σ_max(B₀)` with equality only when `Bq` lies
  in the span of what `B₀` already attains. `σ_max(B₀)` is the deliberate choice:
  it is the scale of the operator actually being inverted, and being the smaller
  of the two it makes the reported defect an upper bound on the `‖B‖₂`-normalised
  one — the diagnostic errs toward flagging, never toward staying silent.
- `pou_error` : `‖M·1 - m‖_∞ / ‖m‖_∞`. The B-splines are a partition of unity,
  so `M·1 = m` identically; this cross-checks the exact `m` against the mass
  matrix independently of the transfer operator.
- `cond_mass`, `cond_mass_scaled` : `κ(M)` and `κ(M_s)`, i.e. before and after
  the diagonal equilibration.
- `sigma_max`, `condition_restricted` : free from the same SVD.

Returns `(norm, sigma_min, sigma_max, condition_restricted, mass_error,
cond_mass, cond_mass_scaled, pou_error, n)`, all scalars in `T`.
"""
function reduced_resolvent_norm(
    M :: AbstractMatrix{T},
    G :: AbstractMatrix{T},
    m :: AbstractVector{T},
) where {T<:AbstractFloat}

    n = size(M, 1)
    f = l2_orthonormal_form(M, G, m)

    # I - P_{ε,N} restricted to the zero-mass space, then a direct dense SVD.
    Q0 = orthonormal_complement(f.q)
    sv = svdvals(adjoint(Q0) * f.B * Q0)

    sigma_min = minimum(sv)
    sigma_max = maximum(sv)

    # q* B = 0 is exact mass preservation.  Normalised by σ_max(B₀), the scale of
    # the operator being inverted — not by σ_max(B), which is generally larger
    # because B may act nontrivially on q itself.  See the docstring.
    mass_error = norm(adjoint(f.B) * f.q) / sigma_max
    mass_error > mass_tolerance(T) && @warn(
        "Galerkin operator does not preserve the zero-mass space to assembly accuracy: " *
        "‖q*B‖/σ_max(B₀) = $mass_error exceeds $(mass_tolerance(T)). The value returned " *
        "here is therefore not the reduced resolvent norm of a mass-preserving operator.")

    # Partition of unity: M·1 = m identically.
    pou_error = maximum(abs, M * ones(T, n) - m) / maximum(abs, m)

    return (
        norm                 = inv(sigma_min),
        sigma_min            = sigma_min,
        sigma_max            = sigma_max,
        condition_restricted = sigma_max / sigma_min,
        mass_error           = mass_error,
        cond_mass            = f.cond_mass,
        cond_mass_scaled     = f.cond_mass_scaled,
        pou_error            = pou_error,
        n                    = n,
    )
end

function reduced_resolvent_norm(
    M       :: AbstractMatrix{T},
    G       :: AbstractMatrix{T},
    splines :: AbstractVector{<:SingleBSpline{T}},
) where {T<:AbstractFloat}
    return reduced_resolvent_norm(M, G, T[ϕ.mass for ϕ in splines])
end

end


"""
    reduced_resolvent_norm(r::InvariantDensityResult) -> NamedTuple

Reduced resolvent norm of a stored run, using its saved `M` and `G` and the
exact integration functional of its rehydrated basis. Nothing is re-assembled.
"""
function ResolventNorm.reduced_resolvent_norm(r::InvariantDensityResult)
    splines = Bases.build_single_splines(rehydrate(r).basis)
    return ResolventNorm.reduced_resolvent_norm(r.M, r.G, splines)
end
