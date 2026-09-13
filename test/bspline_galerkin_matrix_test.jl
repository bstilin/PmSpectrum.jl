# -------------------------------------
# Tests for BSplineGalerkinMatrix.jl
# -------------------------------------
#
# build_star_partition is covered in two layers: SP-1..SP-5 are point-set tests
# on hand-built partitions with known expected members, and SS-0..SS-4 verify
# the construction guarantee by dense sampling on realistic PM meshes.

using Test
using BSplineKit
using BSplineKit: knots as bsk_knots
using DoubleFloats: Double64
using QuadGK: quadgk
using GaussQuadrature: legendre
using PmSpectrum.Bases: SingleBSpline, build_single_splines,
    eval_G_epsilon_piecewise,
    build_star_partition, Branch, EndpointRefinement,
    _endpoint_dyadic_points, default_tau_end,
    mass_matrix, transfer_matrix, bspline_inner_product,
    build_pm_basis, symmetric_rice_breakpoints
using PmSpectrum.Utils: construct_left_branch_inverse, symmetric_pm


######################################################
# eval_G_epsilon_piecewise vs QuadGK
######################################################

@testset "eval_G_epsilon_piecewise vs QuadGK, knot-free and knot-straddling windows (GP-1)" begin
    T = Float64
    break_points = T[0.0, 0.25, 0.5, 0.75, 1.0]
    B    = BSplines.BSplineBasis(BSplineOrder(4), break_points)
    spls = build_single_splines(B)

    for (t, ε) in ((T(0.375), T(0.1)),    # knot-free window
                   (T(0.24),  T(0.05)),   # knot 0.25 inside the window
                   (T(0.5),   T(0.3)))    # several knots inside the window
        for ϕ in spls
            # Skip splines identically zero on the window — trivially true.
            any(a < t + ε && b > t - ε for (a, b) in ϕ.support) || continue

            pw = eval_G_epsilon_piecewise(ϕ, t, ε)

            # QuadGK reference with the interior knots as segment boundaries.
            inner = sort(unique(filter(k -> t - ε < k < t + ε, ϕ.knots)))
            quad_val, _ = quadgk(x -> ϕ(x), t - ε, inner..., t + ε; rtol=1e-14)
            ref = quad_val / ((t + ε) - (t - ε))

            @test pw ≈ ref rtol=1e-12 atol=1e-15
        end
    end
end

@testset "eval_G_epsilon_piecewise wrap cases vs QuadGK (GP-2)" begin
    T = Float64
    break_points = T[0.0, 0.25, 0.5, 0.75, 1.0]
    B    = BSplines.BSplineBasis(BSplineOrder(4), break_points)
    spls = build_single_splines(B)
    ε = T(0.05)

    for t in (T(0.02), T(0.98)), ϕ in spls
        pw = eval_G_epsilon_piecewise(ϕ, t, ε)
        tm, tp = t - ε, t + ε
        ref = if tm < 0     # wrap-left: pieces [0, tp] and [tm+1, 1]
            (quadgk(x -> ϕ(x), zero(T), tp; rtol=1e-14)[1] +
             quadgk(x -> ϕ(x), tm + one(T), one(T); rtol=1e-14)[1]) / (tp - tm)
        else                # wrap-right: pieces [tm, 1] and [0, tp-1]
            (quadgk(x -> ϕ(x), tm, one(T); rtol=1e-14)[1] +
             quadgk(x -> ϕ(x), zero(T), tp - one(T); rtol=1e-14)[1]) / (tp - tm)
        end
        @test pw ≈ ref rtol=1e-12 atol=1e-15
    end
end

@testset "eval_G_epsilon_piecewise small-ε accuracy across a knot (GP-3)" begin
    # Fine mesh near 0; a window of width 2e-9 straddles a knot. The piecewise
    # evaluation must stay at O(eps) relative accuracy — this is the regime
    # where differencing the antiderivative spline lost ~eps(T)·h/(2ε) digits.
    T  = Float64
    bp = vcat(T(0), T.(10.0 .^ range(-8, -1; length=15)), T(0.5), T(1))
    B  = BSplines.BSplineBasis(BSplineOrder(4), copy(bp))
    spls = build_single_splines(B)
    ε  = T(1e-9)

    tested = 0
    for ϕ in spls
        sa, sb = ϕ.support[1]
        kts = unique(ϕ.knots)
        ik  = findfirst(k -> max(sa, 1e-7) < k < sb, kts)
        ik === nothing && continue
        kmid = kts[ik]
        t = kmid + ε / 2                  # window [t-ε, t+ε] contains knot kmid

        pw = eval_G_epsilon_piecewise(ϕ, t, ε)

        inner = sort(unique(filter(k -> t - ε < k < t + ε, kts)))
        quad_val, _ = quadgk(x -> ϕ(x), t - ε, inner..., t + ε; rtol=1e-15)
        ref = quad_val / ((t + ε) - (t - ε))

        @test pw ≈ ref rtol=1e-12
        tested += 1
    end
    @test tested ≥ 3
end


######################################################
# build_star_partition
######################################################

function check_star_partition(part, domain; atol=1e-14)
    da, db = domain
    @test part[1]   ≈ da atol=atol
    @test part[end] ≈ db atol=atol
    @test issorted(part)
    for i in firstindex(part)+1:lastindex(part)
        @test part[i] - part[i-1] > atol / 2
    end
end

function in_star_partition(v, part; atol=1e-12)
    any(p -> abs(p - v) ≤ atol, part)
end

# Each SP testset runs on both the base and the endpoint-refined partition:
# refinement only adds points, so presence/exclusion expectations are identical.
_sp_refines(T) = (nothing, EndpointRefinement(T(0.5)))


# Local helper (test-only): true iff the circular window (t-ε, t+ε) (mod 1)
# contains no knot; the circular distance from t to p is min(|t-p|, 1-|t-p|).
function circular_knot_free(t::T, ε::T, knots::Vector{T}) where {T<:AbstractFloat}
    for p in knots
        d = abs(t - p)
        d = min(d, one(T) - d)
        d <= ε && return false
    end
    return true
end


# The PM branch pair, as the assembly code builds it.  `fwd` must be the
# branch's OWN formula across its closed domain (see SS-0).
function _pm_branches(α::T) where {T<:AbstractFloat}
    left_inv  = construct_left_branch_inverse(α)
    right_inv = x -> one(T) - left_inv(one(T) - x)
    right_fwd = x -> one(T) - symmetric_pm(one(T) - x, α)   # right-branch formula
    b1 = Branch(x -> symmetric_pm(x, α), left_inv,
                (zero(T), T(0.5)), (zero(T), one(T)))
    b2 = Branch(right_fwd, right_inv,
                (T(0.5),  one(T)), (zero(T), one(T)))
    return (b1, b2)
end

@testset "build_star_partition: endpoints, sorted, no duplicates (SP-1)" begin
    T      = Float64
    branch = Branch(identity, identity, (T(0), T(1)), (T(0), T(1)))
    knots  = T[0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
    ε      = T(0.05)

    for refine in _sp_refines(T)
        part = build_star_partition(branch, knots, ε; refine=refine)
        check_star_partition(part, (T(0), T(1)))
    end
end

@testset "build_star_partition: τ± pullbacks present in partition (SP-2)" begin
    T      = Float64
    branch = Branch(identity, identity, (T(0), T(1)), (T(0), T(1)))
    knots  = T[0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
    ε      = T(0.05)

    for refine in _sp_refines(T)
        part     = build_star_partition(branch, knots, ε; refine=refine)
        expected = [0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80]
        for v in expected
            @test in_star_partition(v, part)
        end
    end
end

@testset "build_star_partition: τ± outside subdomain are excluded (SP-3)" begin
    T      = Float64
    branch = Branch(identity, identity, (T(0), T(0.5)), (T(0), T(1)))
    knots  = T[0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
    ε      = T(0.05)

    for refine in _sp_refines(T)
        part = build_star_partition(branch, knots, ε; refine=refine)
        check_star_partition(part, (T(0), T(0.5)))

        for v in [0.55, 0.70, 0.75, 0.80, 0.95]
            @test !in_star_partition(v, part)
        end
        for v in [0.05, 0.20, 0.25, 0.30, 0.45]
            @test in_star_partition(v, part)
        end
    end
end

@testset "build_star_partition: wrapping τ- included (SP-4)" begin
    T      = Float64
    branch = Branch(identity, identity, (T(0), T(1)), (T(0), T(1)))
    knots  = T[0.0, 0.0, 0.0, 0.0, 0.02, 1.0, 1.0, 1.0, 1.0]
    ε      = T(0.05)

    for refine in _sp_refines(T)
        part = build_star_partition(branch, knots, ε; refine=refine)
        check_star_partition(part, (T(0), T(1)))
        @test in_star_partition(T(0.07), part)
        @test in_star_partition(T(0.97), part)
    end
end

@testset "build_star_partition: τ± pulled back through non-identity inverse (SP-5)" begin
    T      = Float64
    fwd    = u -> T(2) * u
    inv_fn = t -> t / T(2)
    branch = Branch(fwd, inv_fn, (T(0), T(0.5)), (T(0), T(1)))
    knots  = T[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    ε      = T(0.05)

    for refine in _sp_refines(T)
        part = build_star_partition(branch, knots, ε; refine=refine)
        check_star_partition(part, (T(0), T(0.5)))
        @test in_star_partition(T(0.275), part)
    end
end


######################################################
# S* property tests on realistic PM meshes
######################################################
#
# The S* construction guarantee (see the paper's transfer-matrix section and
# the build_star_partition docstring): on every subinterval of S*, whether the
# mapped window [T_k(u)-ε, T_k(u)+ε] (mod 1) contains a knot is CONSTANT —
# equivalently, the kink locations of G_ε ϕ ∘ T_k (the pullbacks of knots ± ε)
# are all S* points, never interior to a subinterval. SS-1 verifies that
# invariant by dense sampling, complementing the point-set tests SP-1..SP-5.

@testset "Branch endpoint contract on the PM pair (SS-0)" begin
    # `fwd` must be the branch's OWN formula across its closed domain, so that
    # fwd(domain) lands exactly on range. The right branch cannot just reuse
    # symmetric_pm: that gives x = 1/2 to the LEFT branch, so a shared
    # symmetric_pm closure would return 1 at b2's left endpoint where range[1]
    # is 0 — and _branch_quadrature_data brackets a segment's image from
    # precisely these endpoint values.
    for T in (Float64, Double64), α in (T(0.25), T(0.5), T(0.85))
        b1, b2 = _pm_branches(α)
        for b in (b1, b2)
            @test b.fwd(b.domain[1]) ≈ b.range[1] atol=eps(T)
            @test b.fwd(b.domain[2]) ≈ b.range[2] atol=eps(T)
        end

        # the two branches disagree at the shared breakpoint, by design
        @test b1.fwd(T(0.5)) ≈ one(T)  atol=eps(T)
        @test b2.fwd(T(0.5)) ≈ zero(T) atol=eps(T)

        # away from it they agree with the global map
        for u in (T(0.55), T(0.7), T(0.95))
            @test b2.fwd(u) ≈ symmetric_pm(u, α) rtol=sqrt(eps(T))
        end
    end
end


@testset "S*: structure and knot-window invariance on PM meshes (SS-1)" begin
    T = Float64

    for α in (T(0.15), T(0.3), T(0.7))
        b1, b2 = _pm_branches(α)

        for ε in (T(1e-2), T(1e-4)), n_outer in (6, 12)
            B, _ = build_pm_basis(α, ε, n_outer, T)
            kv   = collect(T, bsk_knots(B))            # full knot vector, as used in assembly

            for branch in (b1, b2), refine in (nothing, EndpointRefinement(α))
                part = build_star_partition(branch, kv, ε; refine=refine)
                da, db = branch.domain

                # Endpoints, ordering, deduplication
                @test part[1] == da
                @test part[end] == db
                @test issorted(part)
                @test all(diff(part) .> eps(T))

                # Every base knot strictly inside the branch domain appears
                for p in unique(kv)
                    da < p < db || continue
                    @test any(x -> abs(x - p) ≤ 4eps(T), part)
                end

                # Invariance: whether the mapped window contains a knot is
                # constant on each subinterval and equals the midpoint
                # classification. Sample strictly inside (5%..95%) to stay
                # clear of the exact window boundaries that S* points sit on.
                for k in 1:length(part)-1
                    ul, ur = part[k], part[k+1]
                    ur - ul ≤ 16eps(T) && continue
                    mid_free = circular_knot_free(branch.fwd((ul + ur)/2), ε, kv)
                    for s in range(T(0.05), T(0.95); length=21)
                        u = ul + s * (ur - ul)
                        @test circular_knot_free(branch.fwd(u), ε, kv) == mid_free
                    end
                end
            end
        end
    end
end


@testset "S*: knot-shift pullbacks present (SS-2)" begin
    # For the left PM branch, every (p ± ε) mod 1 that lands in the branch
    # range must appear in S* as its pullback T_k⁻¹((p ± ε) mod 1).
    T = Float64
    α = T(0.5)
    ε = T(1e-3)
    b1, _ = _pm_branches(α)

    B, _ = build_pm_basis(α, ε, 8, T)
    kv   = collect(T, bsk_knots(B))

    for refine in (nothing, EndpointRefinement(α))
        part = build_star_partition(b1, kv, ε; refine=refine)

        ra, rb = b1.range
        for p in unique(kv), shifted in (mod(p + ε, one(T)), mod(p - ε, one(T)))
            ra ≤ shifted ≤ rb || continue
            u = b1.inv(shifted)
            b1.domain[1] ≤ u ≤ b1.domain[2] || continue
            @test any(x -> abs(x - u) ≤ 1e-9, part)
        end
    end
end


@testset "S*: dyadic endpoint refinement points (SS-3)" begin
    T = Float64

    # A geometric partition with ratio r has h_j = (r-1)·w_j, so with η = 1/2 a
    # panel is graded iff r ≤ 1.5. Ratio 1.4 ⇒ graded everywhere; ratio 1.6 ⇒
    # ungraded everywhere. These give deterministic c_end locations below.

    # Depth: fully graded partition ⇒ the scan stops at the first interior
    # point (c_end = w_1) and the returned points are exactly c_end·2⁻ᵏ,
    # k = 1 … N_end-1, in descending order.
    for (α, τ) in ((T(0.5), T(1e-12)), (T(0.15), T(1e-6)))
        ref   = EndpointRefinement(α; τ_end=τ)
        N_end = ceil(Int, log2(one(T) / τ) / (T(2) + α))

        w1 = T(1e-3)
        P  = vcat(zero(T), [w1 * T(1.4)^k for k in 0:12])   # ratio 1.4: graded
        pts = _endpoint_dyadic_points(P, ref)

        @test length(pts) == N_end - 1
        @test pts ≈ [w1 * T(2.0)^(-k) for k in 1:N_end-1]
        @test maximum(pts) == w1 / 2
        @test minimum(pts) ≈ w1 * T(2.0)^(1 - N_end)
    end

    # c_end scan: ungraded (ratio 1.6) head, graded (ratio 1.4) tail. The first
    # point from which n_check consecutive panels are graded is the last head
    # point (the junction panel to the tail is graded), so c_end lands there.
    let α = T(0.5), ref = EndpointRefinement(α)
        head = T[0.0, 0.01, 0.016, 0.0256]                  # ratio 1.6: ungraded
        tail = [T(0.0256) * T(1.4)^k for k in 1:6]          # ratio 1.4: graded
        P    = vcat(head, tail)
        pts  = _endpoint_dyadic_points(P, ref)
        @test 2 * maximum(pts) ≈ T(0.0256)                  # c_end = junction point
    end

    # Fallback: ungraded everywhere (ratio 1.6) ⇒ no index qualifies and
    # c_end falls back to the last interior point.
    let α = T(0.5), ref = EndpointRefinement(α)
        P   = T[0.0, 0.01, 0.016, 0.0256, 0.04096]
        pts = _endpoint_dyadic_points(P, ref)
        @test 2 * maximum(pts) ≈ T(0.0256)                  # last interior point
    end

    # Degenerate inputs
    let α = T(0.5), ref = EndpointRefinement(α)
        @test isempty(_endpoint_dyadic_points(T[0.0, 0.5], ref))   # no interior point
        @test isempty(_endpoint_dyadic_points(T[0.0], ref))
    end

    # Constructor validation and defaults
    @test_throws ArgumentError EndpointRefinement(T(0.0))
    @test_throws ArgumentError EndpointRefinement(T(1.0))
    @test_throws ArgumentError EndpointRefinement(T(0.5); τ_end=T(0))
    @test_throws ArgumentError EndpointRefinement(T(0.5); τ_end=T(1))
    @test_throws ArgumentError EndpointRefinement(T(0.5); η=T(0))
    @test_throws ArgumentError EndpointRefinement(T(0.5); n_check=0)
    @test default_tau_end(Float64)  == 1e-12
    @test default_tau_end(Double64) == Double64(1e-22)
end


@testset "S*: refinement through build_star_partition (SS-4)" begin
    T = Float64
    atol = eps(T)

    for α in (T(0.15), T(0.5))
        b1, b2 = _pm_branches(α)
        ref    = EndpointRefinement(α)

        for ε in (T(1e-3), T(1e-6)), n_outer in (8, 16)
            B, _ = build_pm_basis(α, ε, n_outer, T)
            kv   = collect(T, bsk_knots(B))

            baseL = build_star_partition(b1, kv, ε)
            refdL = build_star_partition(b1, kv, ε; refine=ref)
            baseR = build_star_partition(b2, kv, ε)
            refdR = build_star_partition(b2, kv, ε; refine=ref)

            # refine=nothing is the default (no-op identity)
            @test build_star_partition(b1, kv, ε; refine=nothing) == baseL

            # Refinement only adds points; every base point survives
            @test length(refdL) > length(baseL)
            @test all(p -> any(x -> abs(x - p) ≤ atol, refdL), baseL)

            # The added points are the dyadic set, modulo atol dedup collisions
            pts = _endpoint_dyadic_points(baseL, ref; atol=atol)
            for x in pts
                @test any(p -> abs(p - x) ≤ atol, refdL) ||
                      any(p -> abs(p - x) ≤ atol, baseL)
            end
            @test length(refdL) - length(baseL) ≤ length(pts)

            # Structure invariants hold for the refined partition
            @test refdL[1] == b1.domain[1] && refdL[end] == b1.domain[2]
            @test issorted(refdL) && all(diff(refdL) .> atol)
            @test refdR[1] == b2.domain[1] && refdR[end] == b2.domain[2]
            @test issorted(refdR) && all(diff(refdR) .> atol)

            # Right branch mirrors the left through 1/2 (symmetric knots and
            # mirrored inverses), up to fp roundoff
            mirror = one(T) .- reverse(refdR)
            @test length(mirror) == length(refdL)
            @test maximum(abs.(mirror .- refdL)) ≤ 1e-12
        end
    end

    # A branch whose domain spans [0, 1] gets dyadic points near both ends
    let α = T(0.5), ε = T(0.05), ref = EndpointRefinement(α)
        branch = Branch(identity, identity, (T(0), T(1)), (T(0), T(1)))
        knots  = T[0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
        base   = build_star_partition(branch, knots, ε)
        refd   = build_star_partition(branch, knots, ε; refine=ref)

        added = filter(x -> all(p -> abs(p - x) > eps(T), base), refd)
        @test any(x -> x < T(0.05), added)          # new points clustered at 0
        @test any(x -> x > T(0.95), added)          # and at 1
    end
end


######################################################
# Mass Matrix
######################################################

@testset "Mass matrix agreement with BSplineKit Galerkin matrix" begin
    T = Float64
    ORDER = 3
    NUM_BREAK_POINTS = 300

    # Rice-type breakpoints and nonperiodic basis
    break_points = symmetric_rice_breakpoints(ORDER - 1, one(T) - T(0.5), NUM_BREAK_POINTS)
    basis = BSplines.BSplineBasis(BSplineOrder(ORDER), break_points)

    M1 = mass_matrix(basis)
    M2 = BSplineKit.Galerkin.galerkin_matrix(basis)

    @test maximum(abs.(M1 - M2)) ≤ 10 * eps(T)
end


######################################################
# transfer_matrix entries vs QuadGK
######################################################

@testset "transfer_matrix entries vs QuadGK reference" begin
    T      = Float64
    α      = T(0.9)
    ε      = T(1e-6)
    DEGREE = 3
    N_BP   = 20

    break_points = symmetric_rice_breakpoints(DEGREE, α, N_BP)
    B    = BSplines.BSplineBasis(BSplineOrder(DEGREE + 1), copy(break_points))
    spls = build_single_splines(B)

    b1, b2   = _pm_branches(α)
    branches = [b1, b2]

    # Independent pointwise evaluator for G_ε gj(t): direct adaptive quadrature
    # over the (possibly wrapped) window, split at the knots it contains. Uses
    # no package evaluation code beyond gj itself.
    function ref_G_eps(gj, t, ε)
        tm, tp = t - ε, t + ε
        seg(a, b) = begin
            inner = sort(unique(filter(k -> a < k < b, gj.knots)))
            quadgk(x -> gj(x), a, inner..., b; rtol=1e-12)[1]
        end
        total = if tm >= zero(T) && tp <= one(T)
            seg(tm, tp)
        elseif tp > one(T)
            seg(tm, one(T)) + seg(zero(T), tp - one(T))
        else
            seg(zero(T), tp) + seg(tm + one(T), one(T))
        end
        return total / (tp - tm)
    end

    # Reference: sum over branches of ∫_{da}^{db} fi(u) * G_ε gj(T_k(u)) du
    function entry_ref(fi, gj)
        total = zero(T)
        for br in branches
            da, db = br.domain
            # Use fi's piece boundaries within the branch domain as quadgk breakpoints
            # so the integrand's kinks don't hurt accuracy.
            pts = T[da]
            for piece in fi.pieces
                piece.a > da && piece.a < db && push!(pts, piece.a)
                piece.b > da && piece.b < db && push!(pts, piece.b)
            end
            push!(pts, db)
            sort!(unique!(pts))

            for k in 1:length(pts)-1
                val, _ = quadgk(pts[k], pts[k+1]; rtol=1e-10) do u
                    fi(u) * ref_G_eps(gj, br.fwd(u), ε)
                end
                total += val
            end
        end
        return total
    end

    n = length(spls)
    test_pairs = [(1, 1), (n÷2, n÷2), (n, n), (n÷2, n÷2 + 1), (2, 3), (n-3, n-1)]
    refs = Dict((j, i) => entry_ref(spls[i], spls[j]) for (j, i) in test_pairs)

    # The refined and unrefined assemblies must both match the same
    # quadrature reference (refinement only subdivides S* panels).
    for refine in (nothing, EndpointRefinement(α))
        G = transfer_matrix(spls, spls, ε, branches; n_quad=16, refine=refine)
        for (j, i) in test_pairs
            @test G[j, i] ≈ refs[(j, i)] rtol=1e-10 atol=0.0
        end
    end
end


######################################################
# bspline_inner_product
######################################################

@testset "bspline_inner_product merge path is exact when knot vectors are identical" begin
    α, ε = 0.5, 0.1
    T = Float64

    basis_a, _ = build_pm_basis(α, ε, 8, T)
    spls_a = build_single_splines(basis_a)
    p = spls_a[1].p
    n_nodes = p + 1
    ξ, ω = legendre(BigFloat, n_nodes)
    nodes, wts = T.(ξ), T.(ω)

    for i in 1:min(5, length(spls_a))
        for j in i:min(i + p, length(spls_a))
            ip_aligned = bspline_inner_product(spls_a[i], spls_a[j], nodes, wts)
            # Pass copies from a rebuilt second basis with same knots to force the merge path
            basis_copy = BSplineBasis(BSplineOrder(p+1), copy(collect(Float64, knots(basis_a))); augment=Val(false))
            spls_copy  = build_single_splines(basis_copy)
            ip_merge   = bspline_inner_product(spls_a[i], spls_copy[j], nodes, wts)
            @test ip_aligned ≈ ip_merge  rtol=1e-13  atol=0.0
        end
    end
end

@testset "bspline_inner_product vs QuadGK reference for bases with the same and different knot vectors" begin
    α, ε = 0.5, 0.1
    T = Float64

    basis, _ = build_pm_basis(α, ε, 10, T)
    spls = build_single_splines(basis)
    n = length(spls)
    p = spls[1].p

    ξ, ω = legendre(BigFloat, p + 1)
    nodes, wts = T.(ξ), T.(ω)

    # Breakpoints for piecewise QuadGK (avoids kinks misleading the adaptive integrator)
    bps = unique!(sort!(filter(x -> zero(T) ≤ x ≤ one(T), collect(T, BSplineKit.knots(basis)))))
    function qgk_ref(ϕi, ϕj)
        total = zero(T)
        for k in 1:length(bps)-1
            val, _ = quadgk(x -> ϕi(x)*ϕj(x), bps[k], bps[k+1]; rtol=1e-13)
            total += val
        end
        return total
    end

    # Same-basis pairs: diagonal and nearest-neighbor entries
    for (i, j) in [(1,1), (n÷2, n÷2), (n,n), (n÷2, n÷2+1), (2, 2+p)]
        @test bspline_inner_product(spls[i], spls[j], nodes, wts) ≈ qgk_ref(spls[i], spls[j])  rtol=1e-13  atol=0.0
    end

    # Cross-basis pairs: basis_b has a different knot vector
    basis_b, _ = build_pm_basis(α, ε, 16, T)
    spls_b = build_single_splines(basis_b)
    nb = length(spls_b)

    bps_b = collect(T, BSplineKit.knots(basis_b))
    bps_union = unique!(sort!(filter(x -> zero(T) ≤ x ≤ one(T), [bps; bps_b])))
    function qgk_ref_cross(ϕi, ϕj)
        total = zero(T)
        for k in 1:length(bps_union)-1
            val, _ = quadgk(x -> ϕi(x)*ϕj(x), bps_union[k], bps_union[k+1]; rtol=1e-13)
            total += val
        end
        return total
    end

    for (i, j) in [(1,1), (n÷2, nb÷2), (n, nb)]
        @test bspline_inner_product(spls[i], spls_b[j], nodes, wts) ≈ qgk_ref_cross(spls[i], spls_b[j])  rtol=1e-11  atol=0.0
    end
end
