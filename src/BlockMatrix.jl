# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

###############################################################################
# EXPERIMENTAL TEST CODE — NOT MATHEMATICALLY VETTED
#
# Everything in this file is exploratory. The block decomposition of the
# Galerkin operator and the projection of the global density approximation have
# not been verified against the analysis, so their output should be treated as
# provisional rather than as a result of the package.
#
# Nothing in the main pipeline depends on it: `run_single_experiment`, the
# Galerkin assembly and the eigensolve never call into this file, and every
# function here is exercised only by `test/block_matrix_test.jl`. Those tests
# check internal consistency — that the block pipeline agrees with the full one
# on a decoupled basis, and that the projection reproduces `rho_hat` pointwise.
# Passing them does not establish that the underlying construction is correct.
#
# Derive and check the mathematics before building on any of this.
###############################################################################

using LinearAlgebra: cholesky, Symmetric, norm

###############################################################################
# Projection of the global density approximation onto a B-spline basis
###############################################################################

"""
    compute_projection_rhs(splines_out, rho_hat, ref_knots, L, cL; ...) -> Vector{T}
    compute_projection_rhs(basis_out,   rho_hat, ref_knots, L, cL; ...) -> Vector{T}

Compute the right-hand-side vector v[j] = ∫₀¹ b_j(x) ρ̂_ε(x) dx for each
output basis function b_j, where ρ̂_ε is the five-region global density
approximation returned by `Utils.construct_global_density_hat`.

The second dispatch accepts a `BSplineBasis` directly and converts it to
`SingleBSpline` wrappers via `build_single_splines` before forwarding.

Each inner product ∫ b_j(x) ρ̂_ε(x) dx is computed by splitting the support of
b_j at the five region boundaries (L, cL, 1-cL, 1-L). In the boundary and blend
regions ([0,L], [L,cL], [1-cL,1-L], [1-L,1]), the integrand is smooth and
`n_quad_smooth`-point GL quadrature is applied per sub-interval. In the central
spline region ([cL, 1-cL]), ρ̂_ε coincides with the reference spline, so the mesh
is refined by the reference basis breakpoints to resolve every polynomial piece;
`n_quad_exact`-point GL quadrature then integrates the polynomial product exactly.

Arguments
---------
- `splines_out`   : `SingleBSpline` wrappers for the output basis, or
- `basis_out`     : `BSplineBasis` converted internally to `SingleBSpline` wrappers.
- `rho_hat`       : callable `x::T -> T` (from `construct_global_density_hat`).
- `ref_knots`     : breakpoint or full knot vector of the reference basis. May contain
                    repeated knots (as in a BSplineKit knot vector) or values outside
                    [0,1]; these are sorted, deduplicated, and clipped to [0,1] internally.
                    Used to form the union mesh in the spline region [cL, 1-cL]
                    for exact polynomial-product integration.
- `L`, `cL`       : boundary-layer scale and matching threshold (cL = c*L).
- `n_quad_smooth` : GL nodes per knot span for smooth/blend regions (default 24).
- `n_quad_exact`  : GL nodes per sub-span in the spline region. Defaults to
                    `ceil((p_out + p_ref + 1)/2)` which is exact for the product.
- `atol`          : sub-interval skip and deduplication tolerance.
"""
function compute_projection_rhs end

function compute_projection_rhs(
    splines_out   :: Vector{<:SingleBSpline{T}},
    rho_hat,
    ref_knots     :: AbstractVector,
    L             :: T,
    cL            :: T;
    n_quad_smooth :: Int = 24,
    n_quad_exact  :: Union{Int,Nothing} = nothing,
    atol          :: T = eps(T),
) :: Vector{T} where {T<:AbstractFloat}

    ref_knots_clean = unique!(sort!(filter(k -> zero(T) ≤ T(k) ≤ one(T), collect(T, ref_knots))))

    p_out = splines_out[1].p

    # Infer p_ref from the reference knots spacing (conservatively assume cubic)
    n_exact = n_quad_exact === nothing ? cld(p_out + 3 + 1, 2) : n_quad_exact

    # Pre-compute GL nodes once (BigFloat → T for precision)
    ξs, ωs = legendre(BigFloat, n_quad_smooth)
    nodes_s, wts_s = T.(ξs), T.(ωs)
    ξe, ωe = legendre(BigFloat, n_exact)
    nodes_e, wts_e = T.(ξe), T.(ωe)

    # Region boundaries (strictly sorted since cL > L)
    region_bds = T[L, cL, one(T) - cL, one(T) - L]

    n = length(splines_out)
    v = zeros(T, n)

    for j in 1:n
        ϕ   = splines_out[j]
        acc = zero(T)

        for pce in ϕ.pieces
            # Split piece [a,b] at any region boundary strictly inside it
            a, b = pce.a, pce.b
            sub_pts = T[a]
            for bd in region_bds
                a + atol < bd < b - atol && push!(sub_pts, bd)
            end
            push!(sub_pts, b)

            for k in firstindex(sub_pts):(lastindex(sub_pts)-1)
                lo, hi = sub_pts[k], sub_pts[k+1]
                hi - lo ≤ atol && continue
                mid = (lo + hi) / T(2)

                if mid ≤ L || mid ≥ one(T) - L
                    # Boundary regions [0,L] and [1-L,1]: U_ε or U_ε(1-x)
                    acc += Utils._gl_integrate(x -> ϕ(x) * rho_hat(x), lo, hi, nodes_s, wts_s)

                elseif mid ≤ cL || mid ≥ one(T) - cL
                    # Blend regions [L,cL] and [1-cL,1-L]: smooth integrand
                    acc += Utils._gl_integrate(x -> ϕ(x) * rho_hat(x), lo, hi, nodes_s, wts_s)

                else
                    # Spline region [cL,1-cL]: refine by reference basis knots
                    # for exact polynomial-product integration
                    inner = filter(t -> lo + atol < t < hi - atol, ref_knots_clean)
                    breaks = merge_breakpoints(T[lo, hi], inner; atol=atol)
                    acc += Utils._gl_integrate_mesh(
                        x -> ϕ(x) * rho_hat(x), breaks, nodes_e, wts_e; atol=atol
                    )
                end
            end
        end

        v[j] = acc
    end

    return v
end

function compute_projection_rhs(
    basis_out     :: BSK.BSplines.AbstractBSplineBasis,
    rho_hat,
    ref_knots     :: AbstractVector,
    L             :: T,
    cL            :: T;
    n_quad_smooth :: Int                  = 24,
    n_quad_exact  :: Union{Int,Nothing}   = nothing,
    atol          :: T                    = eps(T),
) :: Vector{T} where {T<:AbstractFloat}
    compute_projection_rhs(
        build_single_splines(basis_out),
        rho_hat, ref_knots, L, cL;
        n_quad_smooth = n_quad_smooth,
        n_quad_exact  = n_quad_exact,
        atol          = atol,
    )
end


"""
    project_global_approximation(α, ε, basis_out, basis_ref, coeffs_ref;
                                 c_match, rho_half, n_quad_smooth, n_quad_exact, atol)
                                 -> (coeffs::Vector{T}, rho_hat)

Project the global density approximation ρ̂_ε onto `basis_out` by L² projection.
This allows acting on the global approximation with the coefficient operator
and its associated block matrices obatined by Galerkin approximation.
Returns a named tuple `(coeffs, rho_hat)` where `coeffs` is the coefficient vector
c such that ∑ c[j] b_j ≈ ρ̂_ε, and `rho_hat` is the pointwise blended function.

ρ̂_ε blends the closed-form boundary-layer solution U_ε with a numerically
computed low-noise reference density ρ̃_ε (on `basis_ref` with `coeffs_ref`).
For accurate results, `basis_out` should be at least as fine as `basis_ref`.

The projection solves the Gram system M c = v where M[i,j] = (b_i, b_j) (the
mass matrix, already implemented) and v[j] = (b_j, ρ̂_ε) (computed here).

Arguments
---------
- `α`, `ε`      : PM exponent and noise half-width.
- `basis_out`   : output BSplineBasis; coefficient vector lives here.
- `basis_ref`   : BSplineBasis for the reference density ρ̃_ε.
- `coeffs_ref`  : coefficient vector of ρ̃_ε on `basis_ref`.
- `c_match`     : blending constant c > 1 (default 10).
- `rho_half`    : density value at x=1/2; defaults to `S_ref(1/2)`.
- `n_quad_smooth`, `n_quad_exact`, `atol`: forwarded to `compute_projection_rhs`.
"""
function project_global_approximation(
    α           :: T,
    ε           :: T,
    basis_out   :: BSK.BSplines.AbstractBSplineBasis,
    basis_ref   :: BSK.BSplines.AbstractBSplineBasis,
    coeffs_ref  :: AbstractVector;
    c_match       :: Real = 10,
    rho_half      :: Union{Nothing,Real} = nothing,
    n_quad_smooth :: Int = 24,
    n_quad_exact  :: Union{Int,Nothing} = nothing,
    atol          :: T = eps(T),
) where {T<:AbstractFloat}

    # Boundary-layer scale for ε
    L  = Utils.boundary_layer_scale(α, ε)
    cL = T(c_match) * L

    # Wrap the reference spline as a plain callable (no BSplineKit import in Utils)
    S_ref       = BSK.Spline(basis_ref, T.(coeffs_ref))
    ref_density = x -> S_ref(x)

    rho_half_val = rho_half === nothing ? ref_density(one(T) / T(2)) : T(rho_half)

    rho_hat = Utils.construct_global_density_hat(
        α, ε, ref_density;
        c_match = T(c_match), rho_half = rho_half_val,
    )

    splines_out = build_single_splines(basis_out)
    M = mass_matrix(splines_out; atol=atol)
    v = compute_projection_rhs(
        splines_out, rho_hat, collect(T, BSK.knots(basis_ref)), L, cL;
        n_quad_smooth = n_quad_smooth,
        n_quad_exact  = n_quad_exact,
        atol          = atol,
    )

    coeffs = cholesky(Symmetric(M)) \ v
    return (coeffs=coeffs, rho_hat=rho_hat)
end


###############################################################################
# Block partition of a decoupled basis
###############################################################################

"""
    BlockPartition{T<:AbstractFloat}

Partition of a decoupled B-spline basis into three groups determined by the
boundary-layer scale L:

| Field      | Splines whose support is … |
|------------|----------------------------|
| `idx_left` | entirely in [0, L]         |
| `idx_int`  | entirely in [L, 1-L]       |
| `idx_right`| entirely in [1-L, 1]       |

Construct with `partition_basis(splines, L)`. The basis must have been built
with `decouple=true` so that no B-spline support straddles L or 1-L.
"""
struct BlockPartition{T<:AbstractFloat}
    L         :: T
    n         :: Int
    idx_left  :: Vector{Int}
    idx_int   :: Vector{Int}
    idx_right :: Vector{Int}
end

"""
    partition_basis(splines, L; atol) -> BlockPartition{T}

Classify each spline in `splines` into the left boundary ([0,L]), interior
([L,1-L]), or right boundary ([1-L,1]) block by inspecting its `.support`
field. Throws an `ArgumentError` if any spline straddles L or 1-L (which
indicates the basis was not built with `decouple=true`).
"""
function partition_basis(
    splines :: Vector{<:SingleBSpline{T}},
    L       :: T;
    atol    :: T = eps(T),
) :: BlockPartition{T} where {T<:AbstractFloat}

    n         = length(splines)
    idx_left  = Int[]
    idx_int   = Int[]
    idx_right = Int[]

    for (j, s) in enumerate(splines)
        min_sup = minimum(a for (a, _) in s.support)
        max_sup = maximum(b for (_, b) in s.support)

        if max_sup ≤ L + atol
            push!(idx_left, j)
        elseif min_sup ≥ one(T) - L - atol
            push!(idx_right, j)
        elseif min_sup < L - atol && max_sup > L + atol
            throw(ArgumentError(
                "Spline $j (support [$min_sup, $max_sup]) straddles L=$L. " *
                "Build the basis with decouple=true."))
        elseif min_sup < one(T) - L - atol && max_sup > one(T) - L + atol
            throw(ArgumentError(
                "Spline $j (support [$min_sup, $max_sup]) straddles 1-L=$(one(T)-L). " *
                "Build the basis with decouple=true."))
        else
            push!(idx_int, j)
        end
    end

    return BlockPartition{T}(L, n, idx_left, idx_int, idx_right)
end


###############################################################################
# Block decomposition of G and M
###############################################################################

"""
    BlockDecomposition{T<:AbstractFloat}

All nine sub-blocks of the Galerkin matrix G and the three diagonal blocks of
the mass matrix M, indexed by the three spline groups L (left boundary),
Q (interior), and R (right boundary).

Naming convention: `G_XY` has rows in block X and columns in block Y, so it
acts as a linear map **from the Y-block to the X-block**. The off-diagonal mass
blocks are omitted — with `decouple=true` they are exactly zero.

Construct with `decompose_matrices(G, M, partition)`.
"""
struct BlockDecomposition{T<:AbstractFloat}
    partition :: BlockPartition{T}
    G_LL :: Matrix{T};  G_LQ :: Matrix{T};  G_LR :: Matrix{T}
    G_QL :: Matrix{T};  G_QQ :: Matrix{T};  G_QR :: Matrix{T}
    G_RL :: Matrix{T};  G_RQ :: Matrix{T};  G_RR :: Matrix{T}
    M_LL :: Matrix{T};  M_QQ :: Matrix{T};  M_RR :: Matrix{T}
end

"""
    decompose_matrices(G, M, partition; atol) -> BlockDecomposition{T}

Extract all nine sub-blocks of `G` and the three diagonal blocks of `M` using
`partition`. Issues a warning if any off-diagonal block of `M` has ∞-norm
exceeding `atol`, which would indicate that the basis was not decoupled.
"""
function decompose_matrices(
    G         :: Matrix{T},
    M         :: Matrix{T},
    partition :: BlockPartition{T};
    atol      :: T = eps(T),
) :: BlockDecomposition{T} where {T<:AbstractFloat}

    L = partition.idx_left
    Q = partition.idx_int
    R = partition.idx_right

    for (name, blk) in (("M_LQ", M[L, Q]), ("M_LR", M[L, R]), ("M_QR", M[Q, R]))
        n = norm(blk, Inf)
        n > atol && @warn "Non-negligible off-diagonal mass block $name (‖⋅‖∞ = $n). Was the basis built with decouple=true?"
    end

    BlockDecomposition{T}(
        partition,
        G[L,L], G[L,Q], G[L,R],
        G[Q,L], G[Q,Q], G[Q,R],
        G[R,L], G[R,Q], G[R,R],
        M[L,L], M[Q,Q], M[R,R],
    )
end

function _get_G_block(d::BlockDecomposition, row::Symbol, col::Symbol)
    row == :left  && col == :left  && return d.G_LL
    row == :left  && col == :int   && return d.G_LQ
    row == :left  && col == :right && return d.G_LR
    row == :int   && col == :left  && return d.G_QL
    row == :int   && col == :int   && return d.G_QQ
    row == :int   && col == :right && return d.G_QR
    row == :right && col == :left  && return d.G_RL
    row == :right && col == :int   && return d.G_RQ
    row == :right && col == :right && return d.G_RR
    throw(ArgumentError("Block symbols must be :left, :int, or :right; got row=:$row, col=:$col"))
end

function _get_M_block(d::BlockDecomposition, which::Symbol)
    which == :left  && return d.M_LL
    which == :int   && return d.M_QQ
    which == :right && return d.M_RR
    throw(ArgumentError("Block symbol must be :left, :int, or :right; got :$which"))
end

function _get_idx(p::BlockPartition, which::Symbol)
    which == :left  && return p.idx_left
    which == :int   && return p.idx_int
    which == :right && return p.idx_right
    throw(ArgumentError("Block symbol must be :left, :int, or :right; got :$which"))
end


###############################################################################
# Spectral analysis, coefficient restriction/embedding, block application
###############################################################################

"""
    block_spectrum(decomp, which) -> (eigenvalues, eigenvectors)

Solve the generalized eigenvalue problem G_XX v = λ M_XX v for diagonal block
`which ∈ {:left, :int, :right}`, directly at the block's working precision
`T`. Eigenvalues and eigenvectors are sorted by |λ| descending.

Returns `(λs::Vector{Complex{T}}, Vs::Matrix{Complex{T}})` where each column
of `Vs` is a coefficient vector **in the block subspace** (length = block size,
not the full basis). To embed eigenvector `k` into the full basis call
`embed_block_vector(real.(Vs[:, k]), decomp.partition, which)`.
"""
function block_spectrum(
    decomp :: BlockDecomposition{T},
    which  :: Symbol,
) :: Tuple{Vector{Complex{T}}, Matrix{Complex{T}}} where {T<:AbstractFloat}

    G_blk = _get_G_block(decomp, which, which)
    M_blk = _get_M_block(decomp, which)

    λs_raw, Vs_raw = Utils.scaled_nonsymmetric_eigen(G_blk, M_blk)
    p  = sortperm(abs.(λs_raw); rev=true)
    λs = Complex{T}.(λs_raw[p])
    Vs = Complex{T}.(Vs_raw[:, p])

    return λs, Vs
end


"""
    restrict_to_block(c, partition, which) -> SubArray / Vector

Return the sub-vector of `c` corresponding to block `which ∈ {:left, :int, :right}`.
"""
function restrict_to_block(
    c         :: AbstractVector,
    partition :: BlockPartition,
    which     :: Symbol,
)
    return c[_get_idx(partition, which)]
end


"""
    embed_block_vector(v, partition, which) -> Vector{T}

Return a zero vector of length `partition.n` with `v` placed at the indices
of block `which ∈ {:left, :int, :right}` and zeros elsewhere.
"""
function embed_block_vector(
    v         :: AbstractVector{T},
    partition :: BlockPartition{T},
    which     :: Symbol,
) :: Vector{T} where {T<:AbstractFloat}
    out = zeros(T, partition.n)
    out[_get_idx(partition, which)] = v
    return out
end


"""
    apply_block(decomp, c, which_in, which_out) -> Vector{T}

Apply one block of the coefficient-space transfer operator M⁻¹G to `c`.

Restricts `c` to block `which_in`, multiplies by `G[which_out, which_in]`,
then premultiplies by `M[which_out, which_out]⁻¹` to recover the
coefficient-space action. The result is embedded back into a full-length
vector (zeros outside `which_out`).

With a decoupled basis M is block-diagonal, so `(M⁻¹G)_XY = M_XX⁻¹ G_XY`;
this function computes exactly that block, directly at the working precision
`T`. Both `which_in` and `which_out` must be `:left`, `:int`, or `:right`.
"""
function apply_block(
    decomp    :: BlockDecomposition{T},
    c         :: AbstractVector{T},
    which_in  :: Symbol,
    which_out :: Symbol,
) :: Vector{T} where {T<:AbstractFloat}
    c_in    = restrict_to_block(c, decomp.partition, which_in)
    G_block = _get_G_block(decomp, which_out, which_in)
    M_block = _get_M_block(decomp, which_out)
    result  = M_block \ (G_block * c_in)
    return embed_block_vector(result, decomp.partition, which_out)
end
