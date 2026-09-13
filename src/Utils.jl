"""
    symmetric_pm(x, alpha)

Symmetric Pomeau–Manneville map Tₐ : [0,1] → [0,1].

Piecewise definition:
    Tₐ(x) = x + 2^α * x^(1+α),          for 0 ≤ x ≤ 1/2
    Tₐ(x) = x - 2^α * (1 - x)^(1+α),    for 1/2 < x ≤ 1

x = 1/2 belongs to the **left** branch, so Tₐ(1/2) = 1. The right branch's own
formula gives 0 there. On the circle 0 and 1 are the same point, so the dynamics
are unchanged either way, but as representatives on [0,1] they differ — which
matters to anything that reads a branch's endpoint value. A `Branch` therefore
carries its own formula and is evaluated only on its own closed domain; see
`Bases.Branch`.
"""
@inline function symmetric_pm(x::T, alpha::T) where {T<:AbstractFloat}
    (x < zero(T) || x > one(T)) &&
        throw(ArgumentError("x must be in [0,1], got $x"))

    d = min(x, one(T) - x)                   # distance to nearest endpoint
    s = ifelse(x <= T(0.5), one(T), -one(T)) # +1 on left half, -1 on right;
                                             # `<=` gives x = 1/2 to the left branch
    return x + s * exp2(alpha) * d^(one(T) + alpha)
end

@inline symmetric_pm(x::Real, alpha::Real) = symmetric_pm(promote(float(x), float(alpha))...)


"""
    symmetric_pm_derivative(x, alpha)

Derivative Tₐ'(x) of the symmetric Pomeau–Manneville map.

Both branches give the same expression in the distance to the nearest endpoint
d = min(x, 1-x):

    Tₐ'(x) = 1 + (1 + α) * 2^α * d^α

(on the right branch, d/dx[x - 2^α (1-x)^(1+α)] = 1 + (1+α) 2^α (1-x)^α).

The derivative is ≥ 1 everywhere, so `log Tₐ'` needs no absolute value.  It has
a fractional cusp at x = 0 and x = 1 (Tₐ'' ~ d^(α-1) blows up) and a jump at
x = 1/2 for α ≠ 0.
"""
@inline function symmetric_pm_derivative(x::T, alpha::T) ::T where {T<:AbstractFloat}
    (x < zero(T) || x > one(T)) &&
        throw(ArgumentError("x must be in [0,1], got $x"))

    d = min(x, one(T) - x)
    return one(T) + (one(T) + alpha) * exp2(alpha) * d^alpha
end

@inline symmetric_pm_derivative(x::Real, alpha::Real) =
    symmetric_pm_derivative(promote(float(x), float(alpha))...)


"""
    log_symmetric_pm_derivative(x, alpha)

log Tₐ'(x), the observable whose ρ-average is the Lyapunov exponent.

Computed as `log1p((1+α) 2^α d^α)` rather than `log(symmetric_pm_derivative(x, α))`:
near the endpoints the argument is small and the naive form loses relative
precision exactly inside the boundary layer, which is where the high-precision
runs are aimed.
"""
@inline function log_symmetric_pm_derivative(x::T, alpha::T) ::T where {T<:AbstractFloat}
    (x < zero(T) || x > one(T)) &&
        throw(ArgumentError("x must be in [0,1], got $x"))

    d = min(x, one(T) - x)
    return log1p((one(T) + alpha) * exp2(alpha) * d^alpha)
end

@inline log_symmetric_pm_derivative(x::Real, alpha::Real) =
    log_symmetric_pm_derivative(promote(float(x), float(alpha))...)


"""
    symmetric_logspace_grid(N; decades=-16, include_mid=true, include_ends=false, T=Double64)

Construct a symmetric grid of N points in [0,1], clustered near the endpoints 0 and 1,
with optional inclusion of the midpoint 1/2 and/or the endpoints themselves.

Definition:
- The grid is symmetric about x = 1/2.
- Interior points are placed in (0,1/2) and mirrored into (1/2,1).
- The clustering near 0 and 1 is controlled by `decades < 0`:  
  the smallest spacing near the endpoints is approximately 10^decades.

Arguments:
- N :: Integer                : total number of grid points, N ≥ 2
- decades :: Real             : negative value controlling clustering (default -16)
- include_mid :: Bool         : whether to include the midpoint 1/2 (default true)
- include_ends :: Bool        : whether to include endpoints 0 and 1 (default false)
- T :: Type{<:AbstractFloat}  : floating-point type of output points (default Double64)

Returns:
- Vector{T} of length N containing grid points on [0,1].

Errors:
- Throws if the remaining number of interior points is odd (symmetry requires pairs).
"""
function symmetric_logspace_grid(N::Integer;
    decades::Real = -16,
    include_mid::Bool = true,
    include_ends::Bool = false,
    T::Type{<:AbstractFloat} = Double64) :: Vector{T}

    N >= 2      || error("Require N ≥ 2")
    decades < 0 || error("Require decades < 0")

    z = zero(T); o = one(T); half = o / T(2)

    # Points forced by options: [0], [1/2], [1]
    base = (include_ends ? 2 : 0) + (include_mid ? 1 : 0)
    rem  = N - base
    rem >= 0      || throw(ArgumentError("N too small for requested options"))
    iseven(rem)   || throw(ArgumentError("N - base must be even"))
    M = div(rem, 2)  # number of left-half interior points

    # Make M points in (0, 1/2): exponents in (decades, 0), exclude both endpoints
    exps = range(T(decades), T(0); length = M + 2)
    left = half .* exp10.(exps[2:end-1])  # strictly inside (0, 1/2)

    # Assemble: [0], left, [1/2], mirror, [1]
    pts = T[]
    sizehint!(pts, N)
    include_ends && push!(pts, z)
    append!(pts, left)
    include_mid  && push!(pts, half)
    append!(pts, o .- reverse(left))
    include_ends && push!(pts, o)
    return pts
end


"""
    log10_fit_lsq(x, y) -> (m, a10)

Least-squares fit of  log10(y) ≈ a10 + m*log10(x).
Returns slope m and intercept a10. The fitted curve in linear space is
    y ≈ 10^a10 * x^m

Nonpositive / nonfinite (x,y) pairs are filtered out before fitting; a warning
is issued if any points are dropped. At least two *distinct* x-values must
survive the filter — the two-parameter fit is underdetermined otherwise, and a
rank-deficient solve would return a minimum-norm "slope" with no indication that
the data never constrained one.
"""
function log10_fit_lsq(x::AbstractVector, y::AbstractVector)
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length"))

    sel = [isfinite(xi) && isfinite(yi) && xi > 0 && yi > 0 for (xi, yi) in zip(x, y)]
    count(sel) < length(sel) && @warn "log10_fit_lsq: $(length(sel) - count(sel)) point(s) dropped (nonpositive or non-finite)"
    any(sel) || throw(ArgumentError("No positive, finite (x,y) pairs to fit."))
    length(unique(view(x, sel))) ≥ 2 || throw(ArgumentError(
        "Require at least two distinct positive, finite x-values; " *
        "log10(y) ≈ a10 + m*log10(x) is underdetermined otherwise."))

    lx = log10.(view(x, sel))
    ly = log10.(view(y, sel))
    n  = length(lx)
    T  = promote_type(eltype(lx), eltype(ly))

    X = hcat(ones(T, n), lx)  # columns: [1, log10(x)]
    β = X \ ly

    a10 = β[1]; m = β[2]
    return m, a10
end


"""
    scaled_nonsymmetric_eigen(G, M) -> (values, vectors)

Solve the generalized eigenvalue problem  G v = λ M v  where M is symmetric
positive definite and G is not necessarily symmetric.

M is diagonally pre-scaled (so its diagonal becomes all-ones) before Cholesky
factorization, reducing the problem to a standard eigenvalue problem on the
congruence transform  B = L⁻¹ G̃ L⁻ᵀ  where  M̃ = L Lᵀ  is the Cholesky
factorization of the scaled mass matrix. The original eigenvectors are
recovered via  v = D⁻¹ L⁻ᵀ w.

Works with any floating-point type for which `cholesky` and `eigen` are
defined, including `Double64` and `BigFloat`. 'BigFloat'  and 'Double64'
compatibility requires 'GenericLinearAlgebra.jl' and 'GenericSchur.jl' to be
loaded first. 

Arguments
- `G` : n×n matrix (not required to be symmetric)
- `M` : n×n symmetric positive-definite matrix

Returns
- `values`  : length-n vector of eigenvalues (complex in general)
- `vectors` : n×n matrix whose columns are the corresponding right eigenvectors
"""
function scaled_nonsymmetric_eigen(G, M)
    issymmetric(M) ||
        throw(ArgumentError("M must be symmetric"))

    d     = sqrt.(diag(M))
    D_inv = Diagonal(1 ./ d)

    M_s = D_inv * M * D_inv
    G_s = D_inv * G * D_inv

    L = try
        cholesky(Symmetric(M_s)).L
    catch e
        e isa PosDefException &&
            throw(ArgumentError("M must be positive definite"))
        rethrow()
    end

    Lsolve = L \ G_s
    B      = (L \ Lsolve')'

    E    = eigen(B)
    vecs = D_inv * (L' \ E.vectors)
    return E.values, vecs
end


"""
    quadrature_operator_difference(G, Hq, Hqp) -> (D, B)

Measure the difference between two quadrature-discretized operators Hq and
Hqp in the operator norm induced by the Gram matrix G.

G is symmetrized and Cholesky factored, G̃ = L Lᵀ, so that the difference
ΔH = Hqp - Hq can be congruence-transformed into the G-orthonormal basis as
B = L⁻¹ ΔH L⁻ᵀ. No explicit inverse is formed. D = ‖B‖₂ is then the largest
singular value of B, i.e. the operator norm of ΔH under the G inner product.

Works with any floating-point type for which `cholesky` and `svdvals` are
defined, including `Double64` and `BigFloat`. 'BigFloat' compatibility 
requires 'GenericLinearAlgebra.jl'.

Arguments
- `G`   : n×n symmetric positive-definite Gram matrix
- `Hq`  : n×n operator matrix
- `Hqp` : n×n operator matrix, same size as `Hq`

Returns
- `D` : the operator norm ‖L⁻¹ (Hqp - Hq) L⁻ᵀ‖₂
- `B` : the congruence-transformed difference matrix
"""
function quadrature_operator_difference(
    G::AbstractMatrix{T},
    Hq::AbstractMatrix{T},
    Hqp::AbstractMatrix{T},
) where {T<:AbstractFloat}

    size(G, 1) == size(G, 2) ||
        throw(DimensionMismatch("G must be square"))

    size(Hq) == size(G) ||
        throw(DimensionMismatch("Hq must have the same dimensions as G"))

    size(Hqp) == size(G) ||
        throw(DimensionMismatch("Hqp must have the same dimensions as G"))

    # Remove small numerical asymmetry while preserving the element type.
    Gsym = Hermitian((G + adjoint(G)) / T(2))

    # Cholesky factorization G̃ = L Lᵀ.
    F = cholesky(Gsym; check=true)
    L = F.L

    ΔH = Hqp - Hq

    # B = L⁻¹ ΔH L⁻ᵀ. No explicit inverse is formed.
    B = (L \ ΔH) / adjoint(L)

    # D = ‖B‖₂ = largest singular value.
    D = maximum(svdvals(B))

    return D, B
end


"""
    match_nearest(ref_evals, curr_evals) -> Vector{Int}

Greedy nearest-neighbor assignment of the candidate eigenvalues in `curr_evals` to the reference eigenvalues 
`ref_evals`, measuring distance in the complex plane.

The algorithm locks in matches in order of highest confidence. At each step, it finds the absolute smallest distance 
between any unassigned reference eigenvalue and any unclaimed candidate in `curr_evals`, and assigns that pair. 

Returns an index vector `idx` of length `length(ref_evals)`, such that `curr_evals[idx[k]]` is the candidate 
eigenvalue matched to `ref_evals[k]`.
"""
function match_nearest(
    ref_evals  :: AbstractVector{<:Complex},
    curr_evals :: AbstractVector{<:Complex},
)
    m = length(ref_evals)
    n = length(curr_evals)
    n >= m ||
        throw(DimensionMismatch("curr_evals (length $n) must be at least as long as ref_evals (length $m)"))

    idx     = zeros(Int, m)
    claimed = falses(n)

    # Distance from every reference eigenvalue to every candidate.
    dist = [abs(ref_evals[k] - curr_evals[j]) for k in 1:m, j in 1:n]

    # Assign references in order of how confident the match is (smallest best-distance 
    # first), each time claiming the nearest unused candidate.

    for _ in 1:m # for each pair to assign
        best_k = 0
        best_j = 0
        best_d = typemax(eltype(dist))
        for k in 1:m # over all references
            idx[k] == 0 || continue                  # already assigned
            for j in 1:n # over all current
                claimed[j] && continue
                if dist[k, j] < best_d
                    best_d = dist[k, j]
                    best_k = k
                    best_j = j
                end
            end
        end
        idx[best_k]     = best_j
        claimed[best_j] = true
    end

    return idx
end

"""
    track_eigenvalues(spectra; k_track) -> (tracked, idxmap)

Chain-track the leading `k_track` modes across a ladder of spectra ordered
**most-accurate-first** (`spectra[1]` is the reference, e.g. the highest quadrature
run). Column 1 is `spectra[1][1:k_track]`; each subsequent column is matched to the
*previous* column via `match_nearest`, i.e. sequential continuation, which is
robust to large jumps between successive levels.

Every input spectrum is assumed to be sorted by `|λ|` descending (as produced by the
solver), so the naive "`k`-th largest" labeling corresponds to position `k`.

Returns `(tracked, idxmap)`:
- `tracked :: Matrix{Complex{T}}` of size `k_track × L` — the tracked eigenvalue for
  each `(mode, level)`.
- `idxmap :: Matrix{Int}` of size `k_track × L` — the position of that eigenvalue within
  the magnitude-sorted spectrum at each level. `idxmap[k, ℓ] == k` means the tracked
  mode `k` still sits at magnitude rank `k` there.

All spectra must have length `≥ k_track`.
"""
function track_eigenvalues(
    spectra :: AbstractVector{<:AbstractVector{<:Complex}};
    k_track :: Int,
)
    L = length(spectra)
    L >= 1 || throw(ArgumentError("spectra must contain at least one spectrum"))
    k_track >= 1 || throw(ArgumentError("k_track must be positive, got $k_track"))
    for (ℓ, s) in enumerate(spectra)
        length(s) >= k_track ||
            throw(DimensionMismatch("spectrum at level $ℓ has length $(length(s)) < k_track = $k_track"))
    end

    Tc      = complex(float(real(eltype(eltype(spectra)))))
    tracked = Matrix{Tc}(undef, k_track, L)
    idxmap  = Matrix{Int}(undef, k_track, L)

    # Level 1: the reference. Tracked mode k = magnitude rank k.
    for k in 1:k_track
        tracked[k, 1] = spectra[1][k]
        idxmap[k, 1]  = k
    end

    # Subsequent levels: match to the previous (more accurate) column.
    for ℓ in 2:L
        prev = tracked[:, ℓ-1]
        idx  = match_nearest(prev, spectra[ℓ])
        for k in 1:k_track
            tracked[k, ℓ] = spectra[ℓ][idx[k]]
            idxmap[k, ℓ]  = idx[k]
        end
    end

    return tracked, idxmap
end


"""
    order_disagreements(idxmap) -> Vector{Tuple{Int,Int}}

Given the `idxmap` returned by `track_eigenvalues`, return every `(mode, level)`
pair where the tracked ordering departs from the magnitude ordering, i.e. where
`idxmap[k, ℓ] != k`. An empty result means naive `|λ|`-descending labeling agreed with
the tracker everywhere — the sort order was safe to use.
"""
function order_disagreements(idxmap::AbstractMatrix{<:Integer})
    out = Tuple{Int,Int}[]
    K, L = size(idxmap)
    for ℓ in 1:L, k in 1:K
        idxmap[k, ℓ] == k || push!(out, (k, ℓ))
    end
    return out
end


"""
    boundary_layer_scale(α::T, ε::T) -> T

Compute the boundary-layer scale L_ε = ( ε² / 6)^(1/(α+2)) ***when ε is interpreted
as the half width of a uniform noise distribution***. i.e. when the noise is distributed
uniformly in [-ε, ε].
"""
@inline function boundary_layer_scale(α::T, ε::T)::T where {T<:AbstractFloat}
    return (ε^2 / (T(6)))^(inv(α + T(2)))
end


"""
    _gl_integrate(f, a::T, b::T, nodes::Vector{T}, weights::Vector{T}) -> T

Gauss-Legendre quadrature of `f` on `[a, b]` using pre-computed nodes/weights
on `[-1,1]`. Maps via `x = (a+b)/2 + (b-a)/2 * t`.
"""
@inline function _gl_integrate(
    f,
    a::T, b::T,
    nodes::Vector{T}, weights::Vector{T},
) ::T where {T<:AbstractFloat}
    half = (b - a) / T(2)
    cen  = a + half
    acc  = zero(T)
    @inbounds for q in eachindex(nodes)
        acc += weights[q] * f(cen + half * nodes[q])
    end
    return half * acc
end


"""
    _gl_integrate_mesh(f, breaks, nodes, weights; atol) -> T

Composite Gauss-Legendre quadrature of `f` over consecutive intervals defined
by `breaks`. Sub-intervals narrower than `atol` are skipped.
"""
function _gl_integrate_mesh(
    f,
    breaks  :: AbstractVector{T},
    nodes   :: Vector{T},
    weights :: Vector{T};
    atol    :: T = eps(T),
) ::T where {T<:AbstractFloat}
    acc = zero(T)
    @inbounds for k in firstindex(breaks):(lastindex(breaks)-1)
        breaks[k+1] - breaks[k] > atol &&
            (acc += _gl_integrate(f, breaks[k], breaks[k+1], nodes, weights))
    end
    return acc
end
