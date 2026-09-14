# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

"""
    bspline_inner_product(ϕi, ϕj, nodes, weights; atol) -> T

Compute ∫₀¹ ϕi(x) ϕj(x) dx exactly using Gauss-Legendre quadrature.

**Same basis** (`ϕi.basis === ϕj.basis`): Pieces are guaranteed to share endpoints,
 so a single quadrature call per shared breakpoint span suffices.

**Different bases** (different knot vectors, same type T): The piece endpoints of both splines are merged into a single
sorted mesh; GL is applied on each sub-interval of the union.

`nodes` and `weights` must be Gauss-Legendre nodes/weights on [-1,1] in type T,
with `length(nodes) ≥ ceil((ϕi.p + ϕj.p + 1)/2)` for exactness.

Note: All inputs must be in the same float type T; no internal conversion is performed.
"""
function bspline_inner_product(
    ϕi      :: SingleBSpline{T},
    ϕj      :: SingleBSpline{T},
    nodes   :: Vector{T},
    weights :: Vector{T};
    atol    :: T = eps(T),
) ::T where {T<:AbstractFloat}

    length(nodes) == length(weights) ||
        throw(ArgumentError("nodes and weights length mismatch"))

    d    = ϕi.p + ϕj.p
    nmin = cld(d + 1, 2)
    length(nodes) >= nmin ||
        throw(ArgumentError("Insufficient Gauss–Legendre nodes for exactness: " *
                            "need ≥ $nmin for degree d=$d, got $(length(nodes))."))

    return ϕi.basis === ϕj.basis ?
           _bspline_ip_aligned(ϕi, ϕj, nodes, weights; atol=atol) :
           _bspline_ip_merge(ϕi, ϕj, nodes, weights; atol=atol)
end

"""
    _bspline_ip_aligned(ϕi, ϕj, nodes, weights; atol) -> T

Compute ∫₀¹ ϕi(x) ϕj(x) dx for two `SingleBSpline`s from the **same basis**
(`ϕi.basis === ϕj.basis`).

Because both splines share the same knot vector, their polynomial pieces are
guaranteed to have identical breakpoints wherever their supports overlap. The
function walks the two piece lists with a two-pointer merge, matching pieces
by `(a, b)` equality (within `atol`). On each matched span, a single GL
quadrature call integrates ϕi·ϕj exactly. Disjoint spans are skipped by
advancing the pointer whose piece ends earlier. A partial overlap that exceeds
`atol` is a same-basis invariant violation and throws an `ArgumentError`.

Returns zero if the supports are disjoint.
"""
function _bspline_ip_aligned(
    ϕi      :: SingleBSpline{T},
    ϕj      :: SingleBSpline{T},
    nodes   :: Vector{T},
    weights :: Vector{T};
    atol    :: T = eps(T),
) ::T where {T<:AbstractFloat}

    I = ϕi.pieces;  J = ϕj.pieces

    acc = zero(T)
    any_overlap = false
    i = 1; j = 1

    while i ≤ length(I) && j ≤ length(J)
        ai, bi = I[i].a, I[i].b
        aj, bj = J[j].a, J[j].b

        if isapprox(ai, aj; atol=atol) && isapprox(bi, bj; atol=atol)
            any_overlap = true
            mid  = (ai + bi) / 2
            half = (bi - ai) / 2
            @inbounds for q in eachindex(nodes)
                x = mid + half*nodes[q]
                acc += (half*weights[q]) * ϕi(x) * ϕj(x)
            end
            i += 1; j += 1
            continue
        end

        # Clearly disjoint — advance the earlier-ending piece
        if bi ≤ aj + atol
            i += 1
            continue
        elseif bj ≤ ai + atol
            j += 1
            continue
        end

        # Overlapping but endpoints don't match → hard error (same-basis invariant)
        L = max(ai, aj); R = min(bi, bj)
        if (R - L) > atol
            throw(ArgumentError("Found partial-overlap between pieces: " *
                                "I[$i]=($ai,$bi) and J[$j]=($aj,$bj)"))
        end

        if bi < bj || (isapprox(bi, bj; atol=atol) && ai ≤ aj)
            i += 1
        else
            j += 1
        end
    end

    return any_overlap ? acc : zero(T)
end


"""
    _bspline_ip_merge(ϕi, ϕj, nodes, weights; atol) -> T

Compute ∫₀¹ ϕi(x) ϕj(x) dx for two `SingleBSpline`s from **different bases**
(different knot vectors, same float type `T`).

Because the piece boundaries of `ϕi` and `ϕj` do not generally coincide, a
single GL call per piece would span sub-intervals where the integrand changes
polynomial form. Instead, the endpoint lists of both splines are merged
into a single sorted breakpoint mesh via `merge_breakpoints`. On each
sub-interval of the union mesh both ϕi and ϕj are single polynomials, so
composite GL quadrature via `Utils._gl_integrate_mesh` is exact provided
`length(nodes) ≥ ceil((ϕi.p + ϕj.p + 1)/2)`.

The mesh is built from the **piece endpoints** (`ϕ.pieces`) rather than the
parent knot vectors (`ϕ.knots`). This ensure that we do not integrate over
regions on which both splines are identically zero.
"""
function _bspline_ip_merge(
    ϕi      :: SingleBSpline{T},
    ϕj      :: SingleBSpline{T},
    nodes   :: Vector{T},
    weights :: Vector{T};
    atol    :: T = eps(T),
) ::T where {T<:AbstractFloat}
    pts_i = T[ϕi.pieces[1].a; [p.b for p in ϕi.pieces]]
    pts_j = T[ϕj.pieces[1].a; [p.b for p in ϕj.pieces]]
    breaks = merge_breakpoints(pts_i, pts_j; atol=atol)
    return Utils._gl_integrate_mesh(x -> ϕi(x) * ϕj(x), breaks, nodes, weights; atol=atol)
end


"""
    mass_matrix(splines::Vector{SingleBSpline{T}}; atol) -> Matrix{T}
    mass_matrix(basis::AbstractBSplineBasis; atol)       -> Matrix{T}

Assemble the mass matrix M where `M[i,j] = ∫₀¹ bᵢ(x) bⱼ(x) dx`.

Accepts either a pre-built `Vector{SingleBSpline{T}}` or a BSplineKit basis
object (from which `SingleBSpline`s are constructed automatically).

Integration uses `p+1` Gauss-Legendre nodes per span, where `p` is the spline
degree. By the exactness property of GL quadrature, `k` nodes integrate
polynomials of degree up to `2k-1` exactly; with `k = p+1` this covers degree
`2p`, which is the degree of the product bᵢ·bⱼ. The nodes and weights are
generated in `BigFloat` and cast to `T` to avoid precision loss in the
node-placement itself.

Only the upper triangle is computed (exploiting symmetry); off-diagonal entries
beyond the bandwidth `p` are left as zero (B-splines of degree `p` have
overlapping support only within `p` neighbours).

# Arguments
- `splines` / `basis` : the B-spline basis, in either form
- `atol` : tolerance passed to `bspline_inner_product` (default `eps(T)`)

# Returns
Symmetric `Matrix{T}` of size `n × n`, where `n = length(basis)`.

# Notes
Unlike `BSplineKit.galerkin_matrix`, all arithmetic is performed in type `T`
(including `Double64` or `BigFloat`), with no internal conversion to `Float64`.
"""
function mass_matrix end

function mass_matrix(basis::Vector{<:SingleBSpline{T}}; atol::T = eps(T)) where {T<:AbstractFloat}

    n = length(basis)

    p   = basis[1].p
    nlg = p + 1
    ξ, ω = legendre(BigFloat,nlg) 
    ξT, ωT = T.(ξ), T.(ω)

    # Initialize with zeros so all entries outside the band are exactly 0.0
    M = zeros(T, n, n)

    # Fill upper triangle in parallel and mirror
    @threads for i in 1:n
        val = bspline_inner_product(basis[i], basis[i], ξT, ωT; atol=atol)
        M[i,i] = val
        
        # Only compute the non-zero overlap band: up to i + p
        @inbounds for j in (i+1):min(i+p, n)
            v = bspline_inner_product(basis[i], basis[j], ξT, ωT; atol=atol)
            M[i,j] = v
            M[j,i] = v
        end
    end

    return M
end

function mass_matrix(basis::B; atol=nothing) where {B<:BSK.BSplines.AbstractBSplineBasis}
    T = eltype(BSK.knots(basis))

    spls = build_single_splines(basis)

    atolT = atol === nothing ? eps(T) : T(atol)

    return mass_matrix(spls; atol=atolT)
end


###############################################################################
# Branch struct (forward + inverse)
###############################################################################

"""
    Branch{F,Finv,T<:AbstractFloat}

Container for a single strictly monotone branch of the dynamics, holding both
the forward map T_k and its inverse. Increasing and decreasing branches are
both supported.

Fields
------
- `fwd`      :: F              # forward map u ↦ T_k(u)
- `inv`      :: Finv           # inverse map t ↦ T_k^{-1}(t)
- `domain`   :: NTuple{2,T}    # [u_a, u_b] ⊂ [0,1]: pre-image interval
- `range`    :: NTuple{2,T}    # [t_a, t_b] ⊂ [0,1]: image of T_k

Contract
--------
`domain` is **closed**, so adjacent branches of a `BranchSet` share a breakpoint.
`fwd` must implement *this* branch's own formula across the whole closed
`domain`, so that `fwd(domain[1])` and `fwd(domain[2])` are the corresponding
endpoints of `range`. At a shared breakpoint two branches therefore disagree by
design, and neither need agree with the global map — the symmetric PM map, for
instance, gives x = 1/2 to its left branch (`Utils.symmetric_pm(0.5, α) == 1`),
so the right branch must be built from the right-branch formula rather than from
the global map, or its `fwd(0.5)` returns `1` where `range[1] == 0`.
`_branch_quadrature_data` depends on this when bracketing a segment's image.
"""
struct Branch{F,Finv,T<:AbstractFloat}
    fwd     :: F
    inv     :: Finv
    domain  :: NTuple{2,T}
    range   :: NTuple{2,T}
end

# Forward evaluation
@inline function (b::Branch{<:Any,<:Any,T})(u::T) ::T where {T<:AbstractFloat}
    b.fwd(u)
end

"""
    BranchSet{T}

Collection of `Branch{_,_,T}` objects accepted by the assembly routines: either
an `AbstractVector` or a `Tuple`. Prefer a `Tuple` when the branches have
different concrete types (tuple iteration specializes per element).
"""
const BranchSet{T} = Union{AbstractVector{<:Branch{<:Any,<:Any,T}},
                           Tuple{Vararg{Branch{<:Any,<:Any,T}}}}


###############################################################################
# G_ε evaluation
###############################################################################

"""
    eval_G_epsilon_piecewise(ϕ::SingleBSpline{T}, t::T, ε::T, nodes, weights) ::T
    eval_G_epsilon_piecewise(ϕ::SingleBSpline{T}, t::T, ε::T) ::T

Evaluate G_ε ϕ(t) = (1/2ε) ∫_{t-ε}^{t+ε} ϕ(x) dx (mod 1) for B-spline ϕ by splitting the
circular window into its non-wrapping pieces on [0,1] and integrating ϕ
directly via `integrate_bspline` (exact piecewise Gauss-Legendre, split at the
knots of ϕ).

Normalization note: the result is divided by the *represented* window width
`(t+ε) - (t-ε)`, rather than by the ideal width `2ε`. The floating-point
endpoints `fl(t±ε)` each have absolute error of order `eps(T)·t`, so their
actual separation can differ from `2ε` by the same order. Dividing by `2ε`
would therefore introduce a relative normalization error of order
`eps(T)·t/(2ε)`, which becomes large when `ε` is very small. By instead
dividing by the actual represented width—whose subtraction is exact by
Sterbenz's lemma—we average over the interval that is actually represented in
floating point. This removes the `1/ε` amplification and leaves only the error
from the slight displacement of the window, of order `eps(T)·t·|ϕ'|`, which is
negligible on meshes adapted to `ϕ`.

**Preconditions:** 0 < ε < 1/2 (the wrap case analysis relies on it; checked at
the assembly entry points, not here) and t ∈ [0,1].

`nodes`/`weights` is a Gauss-Legendre rule on [-1,1] in `T` with
`length(nodes) ≥ cld(ϕ.p + 1, 2)` (see `integrate_bspline`); the convenience
method without them computes the rule per call.
"""
@inline function eval_G_epsilon_piecewise(
    ϕ       :: SingleBSpline{T},
    t       :: T,
    ε       :: T,
    nodes   :: Vector{T},
    weights :: Vector{T},
) ::T where {T<:AbstractFloat}
    tm = t - ε
    tp = t + ε
    total = if tm >= zero(T) && tp <= one(T)
        integrate_bspline(ϕ, ((tm, tp),), nodes, weights)
    elseif tp > one(T)            # wrap-right; tm ≥ 0 guaranteed since ε < 1/2
        integrate_bspline(ϕ, ((tm, one(T)), (zero(T), tp - one(T))), nodes, weights)
    else                          # wrap-left: tm < 0
        integrate_bspline(ϕ, ((zero(T), tp), (tm + one(T), one(T))), nodes, weights)
    end
    return total / (tp - tm)      # actual represented width, not 2ε (see docstring)
end

function eval_G_epsilon_piecewise(ϕ::SingleBSpline{T}, t::T, ε::T) ::T where {T<:AbstractFloat}
    ξ, ω = legendre(BigFloat, cld(ϕ.p + 1, 2))
    return eval_G_epsilon_piecewise(ϕ, t, ε, T.(ξ), T.(ω))
end



###############################################################################
# S* Partition
###############################################################################

"""
    default_tau_end(::Type{T}) :: T

Default endpoint-refinement target `τ_end` for the given working precision.

`τ_end` controls the dyadic refinement depth near a neutral endpoint; it is
not an absolute quadrature-error tolerance.
"""
default_tau_end(::Type{Float64})  = 1e-12
default_tau_end(::Type{Double64}) = Double64(1e-22)


"""
    EndpointRefinement(α::T; τ_end = default_tau_end(T), η = T(1)/2, n_check = 3)

Parameters controlling the optional neutral-endpoint refinement in
`build_star_partition`. See that function for the construction and motivation.

Fields
------
- `α` :
    PM-map exponent.

- `τ_end` :
    Endpoint-refinement target controlling the dyadic depth. Smaller values
    add more levels. This is not an absolute quadrature-error tolerance.
    Default: `default_tau_end(T)`.

- `η` :
    Local-gradedness threshold used to determine where the existing mesh
    becomes sufficiently fine. Default: `1/2`.

- `n_check` :
    Number of successive available panels checked when deciding that the
    existing mesh is sufficiently graded. Default: `3`.
"""
struct EndpointRefinement{T<:AbstractFloat}
    α       :: T
    τ_end   :: T
    η       :: T
    n_check :: Int
end

function EndpointRefinement(
    α::T;
    τ_end::T = default_tau_end(T),
    η::T = T(1)/2,
    n_check::Int = 3,
) where {T<:AbstractFloat}

    0 < α < 1      || throw(ArgumentError("Require 0 < α < 1, got α=$α"))
    0 < τ_end < 1  || throw(ArgumentError("Require 0 < τ_end < 1, got τ_end=$τ_end"))
    0 < η < 1      || throw(ArgumentError("Require 0 < η < 1, got η=$η"))
    n_check ≥ 1    || throw(ArgumentError("Require n_check ≥ 1, got n_check=$n_check"))

    EndpointRefinement{T}(α, τ_end, η, n_check)
end


"""
    build_star_partition(branch::Branch{F,Finv,T}, base_knots::Vector{T}, ε::T;
                         atol::T = eps(T),
                         refine::Union{Nothing,EndpointRefinement{T}} = nothing) :: Vector{T}

Build the partition `S*` used for composite Gauss-Legendre integration of

    f_i(u) * (G_ε g_j)(T_k(u))

over the domain of `branch`.

The partition resolves the B-spline and convolution breakpoints and, when
`refine` is supplied, adds dyadic refinement near neutral endpoints where the
base mesh stops at approximately the boundary-layer scale `L_ε`.

See extended help for the endpoint-refinement motivation and construction.

# Extended help

## Endpoint refinement

After splitting at the B-spline knots and at the pullbacks of the
`±ε`-shifted convolution breakpoints, `f_i` and `G_ε g_j` are polynomial on
each panel. Near a neutral endpoint, in the local coordinate `x` measuring
distance from that endpoint, the PM branch has the form

    T_k(x) = x + 2^α * x^(1 + α),

with the reflected analogue at the right endpoint. Taylor expansion of the
convolution factor therefore gives a polynomial part plus a leading
nonpolynomial correction of order `x^(1 + α)`. Gauss-Legendre resolves the
polynomial part, while the integral of the correction over an endpoint panel
of width `h` scales like

    h^(2 + α).

The base B-spline mesh is already strongly graded toward each neutral endpoint,
down to approximately the boundary-layer scale `L_ε`. Endpoint refinement
therefore supplements only the innermost region, roughly `[0, L_ε]` in the
local endpoint coordinate.

The outer edge `c_end` is determined from the existing mesh. Moving outward
from the endpoint, the mesh is regarded as sufficiently graded once the next
`n_check` available panels satisfy

    h_j ≤ η * p_j,

where `p_j` is the distance of the panel from the endpoint and `h_j` its
width.

Starting from the scale `c_end`, after `N` dyadic halvings the innermost scale
is

    c_end * 2^(-N).

Since the leading endpoint contribution scales like `h^(2 + α)`, its size at
this scale is

    O(c_end^(2 + α) * 2^(-N * (2 + α))).

Thus `2^(-N * (2 + α))` is the reduction relative to the contribution at the
outer scale `c_end`. The dyadic depth `N_end` is chosen as the smallest integer
satisfying

    2^(-N_end * (2 + α)) ≤ τ_end.

Hence `τ_end` is a relative endpoint-refinement target, not an absolute bound
on the total quadrature error.

## Construction

1. Add the branch endpoints and the B-spline knots lying in the branch domain.

2. Add the pullbacks through `branch.inv` of the wrapped `±ε` shifts of the
   B-spline knots, then sort and deduplicate.

3. If `refine` is supplied, determine `c_end` at each neutral endpoint and add
   the corresponding ratio-2 dyadic grid

       c_end / 2, c_end / 4, ..., c_end / 2^(N_end - 1).

   The right endpoint is handled in the reflected coordinate `1 - u`.

4. Sort and deduplicate once more.

The dyadic points resolve the neutral-endpoint behavior itself; they are not
convolution breakpoints and are therefore not passed through the
`±ε` shift-and-pullback construction.
"""
function build_star_partition(
    branch     :: Branch{<:Any,<:Any,T},
    base_knots :: Vector{T},
    ε          :: T;
    atol       :: T = eps(T),
    refine     :: Union{Nothing,EndpointRefinement{T}} = nothing,
) ::Vector{T} where {T<:AbstractFloat}

    da, db = branch.domain
    ra, rb = branch.range
    rmin, rmax = min(ra, rb), max(ra, rb)

    breaks = T[]
    sizehint!(breaks, 4 * length(base_knots) + 4)

    # Domain endpoints are mandatory.
    push!(breaks, da, db)

    # Sorted unique knots in [0,1]. `base_knots` is already sorted.
    knots01 = T[]
    sizehint!(knots01, length(base_knots))

    prev = typemin(T)
    for p in base_knots
        if zero(T) <= p <= one(T) && p - prev > atol
            push!(knots01, p)
            prev = p
        end
    end

    # Breakpoints of f_i inside the branch domain.
    for p in knots01
        if da - atol <= p <= db + atol
            push!(breaks, clamp(p, da, db))
        end
    end

    # Pull back the ±ε-shifted breakpoints of G_ε g_j.
    for p in knots01
        for shifted in (mod(p + ε, one(T)), mod(p - ε, one(T)))
            if rmin - atol <= shifted <= rmax + atol
                u = branch.inv(clamp(shifted, rmin, rmax))

                if da - atol <= u <= db + atol
                    push!(breaks, clamp(u, da, db))
                end
            end
        end
    end

    P = _sorted_dedup!(breaks, atol)
    refine === nothing && return P

    # Construct both endpoint grids from the same sorted structural partition.
    # The right endpoint is handled in the reflected coordinate 1 - u.
    left_pts = da <= atol ?
        _endpoint_dyadic_points(P, refine; atol=atol) :
        T[]

    right_pts = db >= one(T) - atol ?
        one(T) .- _endpoint_dyadic_points(
            reverse(one(T) .- P),
            refine;
            atol=atol,
        ) :
        T[]

    append!(P, left_pts)
    append!(P, right_pts)

    return _sorted_dedup!(P, atol)
end


"""
    _sorted_dedup!(v::Vector{T}, atol::T) :: Vector{T}

Sort `v` in place and return a new vector whose successive retained points
differ by more than `atol`. `v` must be non-empty.
"""
function _sorted_dedup!(v::Vector{T}, atol::T) ::Vector{T} where {T<:AbstractFloat}
    sort!(v)

    out = T[v[1]]
    for i in 2:length(v)
        if v[i] - out[end] > atol
            push!(out, v[i])
        end
    end

    return out
end


"""
    _endpoint_dyadic_points(P_local::Vector{T}, ref::EndpointRefinement{T};
                            atol::T = eps(T)) :: Vector{T}

Return the additional dyadic points for one neutral endpoint.

`P_local` is the sorted partition in the local coordinate measuring distance
from the endpoint, with `P_local[1] == 0`.

The routine finds the outer refinement point `c_end` using the local
gradedness test `h_j ≤ ref.η * w_j`, applied to `ref.n_check` consecutive
panels (truncated at the end of the partition); if no breakpoint passes,
`c_end` falls back to the last interior point. It then inserts

    c_end / 2, ..., c_end / 2^(N_end - 1),

where `N_end` is determined by `ref.τ_end`. Points at or below `atol` are
discarded, as are all deeper ones. Returns an empty vector when `P_local` has
no interior point.
"""
function _endpoint_dyadic_points(
    P_local :: Vector{T},
    ref     :: EndpointRefinement{T};
    atol    :: T = eps(T),
) ::Vector{T} where {T<:AbstractFloat}

    n = length(P_local)
    n ≥ 3 || return T[]

    # Interior points w_j = P_local[j+1]; following panel widths
    # h_j = P_local[j+2] - P_local[j+1].
    n_panels = n - 2

    # If the mesh never passes the gradedness test, refine out to the last
    # interior partition point.
    c_end = P_local[n-1]

    for J in 1:n_panels
        graded = true

        for j in J:min(J + ref.n_check - 1, n_panels)
            w_j = P_local[j+1]
            h_j = P_local[j+2] - P_local[j+1]

            if h_j > ref.η * w_j
                graded = false
                break
            end
        end

        if graded
            c_end = P_local[J+1]
            break
        end
    end

    N_end = ceil(
        Int,
        log2(one(T) / ref.τ_end) / (T(2) + ref.α),
    )

    pts = T[]
    sizehint!(pts, max(N_end - 1, 0))

    x = c_end
    for _ in 1:N_end-1
        x /= 2

        # All later dyadic points are smaller.
        x > atol || break

        push!(pts, x)
    end

    return pts
end

###############################################################################
# Galerkin Matrix assembly
###############################################################################

"""
    transfer_matrix(
        fs::Vector{<:SingleBSpline{T}},
        gs::Vector{<:SingleBSpline{T}},
        ε::T,
        branches;   # Tuple or AbstractVector of Branch{_,_,T}
        n_quad::Int = 32,
        atol::T = eps(T),
        refine::Union{Nothing,EndpointRefinement{T}} = nothing,
    ) ::Matrix{T}

    transfer_matrix(
        basis::BSK.BSplines.AbstractBSplineBasis,
        ε::Tb,
        branches;
        n_quad::Int = 32,
        atol::Tb = eps(Tb),
        refine::Union{Nothing,EndpointRefinement{Tb}} = nothing,
    ) ::Matrix{Tb}

Assemble the Galerkin matrix M[j,i] = (G_εP f_i, g_j) via the identity

    (G_εP f_i, g_j) = ∑_k ∫_{dom(T_k)} f_i(u) · G_ε g_j(T_k(u)) du.

S* partitions are precomputed once per branch (independent of i and j) and
reused across all matrix entries.

Arguments (first method)
------------------------
- `fs`       : trial functions f_i (column index)
- `gs`       : test functions g_j (row index)
- `ε`        : noise half-width; must satisfy 0 < ε < 0.5
- `branches` : branches of the map (each a `Branch{_,_,T}`), must be strictly monotone.
               Prefer a `Tuple` over a `Vector` when the branches have different
               concrete types (e.g. different inverse callables): tuple iteration
               specializes per element and avoids dynamic dispatch in the
               quadrature loop.
- `n_quad`   : number of Gauss-Legendre nodes per subinterval of S*
- `atol`     : partition deduplication and span-skip tolerance
- `refine`   : optional `EndpointRefinement` adding a dyadic grid to S* near the
               integrand singularities at 0 and 1 (see `build_star_partition`)

Returns
-------
`Matrix{T}` of size ng × nf with `M[j,i] = (G_εP f_i, g_j)`.


Design note — shared knot assumption
--------------------------------------
`build_star_partition` is called with `fs[1].knots` as the representative knot
vector. This is correct only when every spline in both `fs` and `gs` shares the
same basis (and therefore the same knot vector). Basis identity is enforced at
the start of the function via `===` checks on `.basis`.
"""
function transfer_matrix end

function transfer_matrix(
    basis    :: B,
    ε        :: Tb,
    branches :: BranchSet{Tb};
    n_quad   :: Int = 32,
    atol     :: Tb  = eps(Tb),
    refine   :: Union{Nothing,EndpointRefinement{Tb}} = nothing,
) where {B<:BSK.BSplines.AbstractBSplineBasis, Tb<:AbstractFloat}

    Tbasis = eltype(BSK.knots(basis))
    Tb === Tbasis ||
        throw(ArgumentError("basis scalar type is $Tbasis but branches have type $Tb"))

    spls = build_single_splines(basis)
    return transfer_matrix(spls, spls, ε, branches; n_quad=n_quad, atol=atol, refine=refine)
end


###############################################################################
# Assembly internals — precomputed per-branch quadrature data
###############################################################################

"""
    _branch_quadrature_data(branch, knots, ε, nodes; atol, refine=nothing) -> NamedTuple

Precompute everything about one branch that is independent of the matrix
indices (i, j): the S* partition, per-subinterval validity flags, the
quadrature nodes `u` in the integration domain, their images `t = T_k(u)`,
and the image range `[tlo, thi]` of each subinterval.

Fields of the returned NamedTuple
---------------------------------
- `breaks` : S* partition of the branch domain (from `build_star_partition`)
- `keep`   : `keep[k]` is false for subintervals narrower than `atol` (skipped)
- `halves` : half-widths `(breaks[k+1] - breaks[k])/2`
- `us`     : `nq × nseg` matrix of quadrature nodes in u
- `ts`     : `nq × nseg` matrix of mapped nodes `T_k(u)`
- `tlo`, `thi` : image range of each subinterval (handles decreasing branches)
"""
function _branch_quadrature_data(
    branch :: Branch{<:Any,<:Any,T},
    knots  :: Vector{T},
    ε      :: T,
    nodes  :: Vector{T};
    atol   :: T = eps(T),
    refine :: Union{Nothing,EndpointRefinement{T}} = nothing,
) where {T<:AbstractFloat}

    breaks = build_star_partition(branch, knots, ε; atol=atol, refine=refine)
    nseg   = length(breaks) - 1
    nq     = length(nodes)

    keep   = falses(nseg)
    halves = zeros(T, nseg)
    us     = zeros(T, nq, nseg)
    ts     = zeros(T, nq, nseg)
    tlo    = zeros(T, nseg)
    thi    = zeros(T, nseg)

    @inbounds for k in 1:nseg
        ul, ur = breaks[k], breaks[k+1]
        ur - ul <= atol && continue   # keep[k] stays false; segment is skipped
        keep[k]   = true
        half      = (ur - ul) / T(2)
        umid      = (ul + ur) / T(2)
        halves[k] = half
        for q in 1:nq
            u        = umid + half * nodes[q]
            us[q, k] = u
            ts[q, k] = branch.fwd(u)
        end
        # Endpoint values, so this relies on `fwd` being the branch's own formula
        # across its closed domain (see `Branch`): a branch whose `fwd` is the
        # global map returns the neighbouring branch's value at a shared
        # breakpoint, and the bracket then misses the segment's true image.
        tlo[k], thi[k] = minmax(branch.fwd(ul), branch.fwd(ur))
    end

    return (breaks=breaks, keep=keep, halves=halves,
            us=us, ts=ts, tlo=tlo, thi=thi)
end


"""
    _fi_quadrature_cache(fs, data, weights) -> Vector{@NamedTuple{ks, wfi}}

For each trial spline `f_i`, precompute the S* subinterval indices `ks` whose
interior overlaps `supp(f_i)` and the weighted node values
`wfi[q, idx] = (half_k · ω_q) · f_i(u_{q,k})` on those subintervals. These are
independent of the test index j and reused across the whole row loop.
"""
function _fi_quadrature_cache(
    fs      :: Vector{<:SingleBSpline{T}},
    data,
    weights :: Vector{T},
) where {T<:AbstractFloat}

    nq     = size(data.us, 1)
    nseg   = length(data.keep)
    breaks = data.breaks

    return map(fs) do fi
        ks = Int[]
        @inbounds for k in 1:nseg
            data.keep[k] && Utils.overlaps_any(breaks[k], breaks[k+1], fi.support) &&
                push!(ks, k)
        end
        wfi = Matrix{T}(undef, nq, length(ks))
        @inbounds for (idx, k) in enumerate(ks)
            h = data.halves[k]
            for q in 1:nq
                wfi[q, idx] = (h * weights[q]) * fi(data.us[q, k])
            end
        end
        (ks = ks, wfi = wfi)
    end
end


"""
    _window_touches_support(tlo, thi, ε, support) -> Bool

Return `true` if the union of circular windows `[t-ε, t+ε] (mod 1)` over
`t ∈ [tlo, thi]` can intersect any interval in `support` (a vector of
non-wrapping `(a, b)` tuples in [0,1]). When false, `G_ε ϕ(t)` is identically
zero on the whole subinterval and its evaluation can be skipped.
"""
@inline function _window_touches_support(tlo::T, thi::T, ε::T, support) where {T<:AbstractFloat}
    a = tlo - ε
    b = thi + ε
    Utils.overlaps_any(a, b, support) && return true
    a < zero(T) && Utils.overlaps_any(a + one(T), one(T), support) && return true   # wrap-left piece
    b > one(T) && Utils.overlaps_any(zero(T), b - one(T), support) && return true   # wrap-right piece
    return false
end


function transfer_matrix(
    fs       :: Vector{<:SingleBSpline{T}},
    gs       :: Vector{<:SingleBSpline{T}},
    ε        :: T,
    branches :: BranchSet{T};
    n_quad   :: Int = 32,
    atol     :: T   = eps(T),
    refine   :: Union{Nothing,EndpointRefinement{T}} = nothing,
) ::Matrix{T} where {T<:AbstractFloat}

    (zero(T) < ε < one(T)/T(2)) ||
        throw(ArgumentError("ε must be in (0, 0.5), got $ε"))

    all(s.basis === fs[1].basis for s in fs) ||
        throw(ArgumentError("all splines in fs must come from the same basis"))
    all(s.basis === gs[1].basis for s in gs) ||
        throw(ArgumentError("all splines in gs must come from the same basis"))
    fs[1].basis === gs[1].basis ||
        throw(ArgumentError("fs and gs must come from the same basis"))

    # Validate the shared knot vector once here; `integrate_bspline` relies on
    # sorted knots for its searchsorted piece-splitting.
    issorted(fs[1].knots) ||
        throw(ArgumentError("knot vector must be sorted"))

    nf = length(fs)
    ng = length(gs)

    # GL nodes/weights: generated in BigFloat once, cast to T
    ξB, ωB  = legendre(BigFloat, n_quad)
    nodes   = T.(ξB)
    weights = T.(ωB)

    # Minimal exact rule for integrating a single degree-p spline, used by
    # eval_G_epsilon_piecewise at every mapped node (see integrate_bspline).
    nG = cld(gs[1].p + 1, 2)
    ξG, ωG   = legendre(BigFloat, nG)
    gnodes   = T.(ξG)
    gweights = T.(ωG)

    # Everything independent of (i, j), once per branch; everything independent
    # of j, once per (branch, i). `map` over a Tuple of branches keeps the
    # per-branch data concretely typed.
    knots = fs[1].knots
    datas = map(br -> _branch_quadrature_data(br, knots, ε, nodes; atol=atol, refine=refine), branches)
    fcs   = map(d -> _fi_quadrature_cache(fs, d, weights), datas)

    M  = zeros(T, ng, nf)
    nq = n_quad

    Threads.@threads for j in 1:ng
        gj  = gs[j]
        sup = gj.support
        for (d, fc) in zip(datas, fcs)
            nseg   = length(d.keep)
            gjvals = Matrix{T}(undef, nq, nseg)
            rel    = falses(nseg)
            @inbounds for k in 1:nseg
                d.keep[k] || continue
                _window_touches_support(d.tlo[k], d.thi[k], ε, sup) || continue
                rel[k] = true
                for q in 1:nq
                    gjvals[q, k] = eval_G_epsilon_piecewise(gj, d.ts[q, k], ε, gnodes, gweights)
                end
            end
            @inbounds for i in 1:nf
                ks  = fc[i].ks
                wfi = fc[i].wfi
                s   = zero(T)
                for (idx, k) in enumerate(ks)
                    rel[k] || continue
                    for q in 1:nq
                        s += wfi[q, idx] * gjvals[q, k]
                    end
                end
                M[j, i] += s
            end
        end
    end

    return M
end
