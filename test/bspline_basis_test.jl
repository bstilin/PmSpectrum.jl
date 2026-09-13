# -------------------------------------
# Tests for BSplineBasis.jl
# -------------------------------------


using BSplineKit
using DoubleFloats
using Random
using PmSpectrum.Bases: symmetric_rice_breakpoints, hybrid_equidistribution_breakpoints,
    SingleBSpline, build_single_splines, l1_norm_difference,
    build_pm_basis, convert_precision, merge_breakpoints, integrate_bspline
using PmSpectrum.Utils: boundary_layer_scale
using Roots: find_zeros
using GaussQuadrature: legendre
using QuadGK: quadgk


######################################################
# integrate_bspline
######################################################

@testset "integrate_bspline: exact single-interval integrals vs QuadGK (IB-1)" begin
    T = Float64
    break_points = T[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    B    = BSplineKit.BSplineBasis(BSplineOrder(4), copy(break_points))
    spls = build_single_splines(B)

    for ϕ in spls
        sa, sb = ϕ.support[1]

        cases = [
            (sa + (sb - sa)/8, sa + (sb - sa)/6),   # inside a single piece
            (sa + (sb - sa)/8, sb - (sb - sa)/8),   # spans several knots
            (max(zero(T), sa - T(0.05)), min(one(T), sb + T(0.05))),  # clipped by support
        ]
        for (a, b) in cases
            a < b || continue
            inner = sort(unique(filter(k -> a < k < b, ϕ.knots)))
            ref, _ = quadgk(x -> ϕ(x), a, inner..., b; rtol=1e-14)
            @test integrate_bspline(ϕ, a, b) ≈ ref rtol=1e-13 atol=1e-16
        end

        # full domain equals the stored mass
        @test integrate_bspline(ϕ, zero(T), one(T)) ≈ ϕ.mass rtol=1e-13

        # degenerate and disjoint intervals are exactly zero
        mid = (sa + sb) / 2
        @test integrate_bspline(ϕ, mid, mid) == zero(T)
        if sb < T(0.9)
            @test integrate_bspline(ϕ, T(0.95), one(T)) == zero(T)
        end
    end
end

@testset "integrate_bspline: multi-interval dispatches and explicit rule (IB-2)" begin
    T = Float64
    break_points = T[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    B    = BSplineKit.BSplineBasis(BSplineOrder(4), copy(break_points))
    spls = build_single_splines(B)
    ϕ    = spls[4]

    ivs_t = ((T(0.1), T(0.3)), (T(0.5), T(0.7)))
    ivs_v = [iv for iv in ivs_t]
    expected = integrate_bspline(ϕ, T(0.1), T(0.3)) + integrate_bspline(ϕ, T(0.5), T(0.7))

    @test integrate_bspline(ϕ, ivs_t) ≈ expected rtol=1e-14
    @test integrate_bspline(ϕ, ivs_v) ≈ expected rtol=1e-14

    # explicit precomputed rule matches the convenience path
    ξ, ω = legendre(BigFloat, cld(ϕ.p + 1, 2))
    nodes, weights = T.(ξ), T.(ω)
    @test integrate_bspline(ϕ, ivs_t, nodes, weights) == integrate_bspline(ϕ, ivs_t)

    # argument validation
    @test_throws ArgumentError integrate_bspline(ϕ, T(0.3), T(0.1))
    @test_throws ArgumentError integrate_bspline(ϕ, T(-0.1), T(0.5))
    @test_throws ArgumentError integrate_bspline(ϕ, T(0.5), T(1.1))
    @test_throws ArgumentError integrate_bspline(ϕ, T(0.1), T(0.3), nodes[1:1], weights[1:1])
end

@testset "integrate_bspline: stacked-knot (decoupled) basis (IB-3)" begin
    T = Float64
    B, _ = build_pm_basis(T(0.5), T(1e-4), 8, T; decouple=true)
    spls = build_single_splines(B)
    for ϕ in spls[1:3:end]
        sa, sb = ϕ.support[1]
        a = max(zero(T), sa - T(0.01))
        b = min(one(T),  sb + T(0.01))
        inner = sort(unique(filter(k -> a < k < b, ϕ.knots)))
        ref, _ = quadgk(x -> ϕ(x), a, inner..., b; rtol=1e-14)
        @test integrate_bspline(ϕ, a, b) ≈ ref rtol=1e-12 atol=1e-16
    end
end


######################################################
# Tests for SingleBSpline
######################################################

# In this test we evaluate a SingleBSpline at various points using the BSplineKit
# evaluation machinery, and compare against evaluating the piecewise polynomial
# expansion (in centered-monomial form) on each piece. This tests that the polynomial
# pieces have been constructed correctly.

@testset "SingleBSpline piecewise polynomial matches centered-monomial expansion" begin
    T = Float64
    ORDER = 4
    NUM_BREAK_POINTS = 200

    # Rice-type breakpoints and nonperiodic basis (your construction)
    break_points = symmetric_rice_breakpoints(ORDER - 1, one(T) - T(0.5), NUM_BREAK_POINTS)
    basis = BSplines.BSplineBasis(BSplineOrder(ORDER), break_points)
    spls  = build_single_splines(basis)

    # Helper: evaluate polynomial from centered-monomial coeffs on a piece
    function poly_eval_on_piece(ϕ, piece, x)::T
        # piece.coeffs is NTuple{P+1,T}, with P == ϕ.p
        dx = x - piece.x0
        s  = zero(T)

        @inbounds for k in 0:ϕ.p
            s += piece.coeffs[k+1] * (dx^k)
        end
        return s
    end

    # Tolerances (Float64-safe)
    rtol = T(1e3) * eps(T)
    atol = eps(T)

    for j in 1:length(basis)
        ϕ = SingleBSpline(basis, j)

        # Each piece is polynomial; test multiple points in [a,b]
        for piece in ϕ.pieces
            a, b = piece.a, piece.b
            @test a < b  # sanity

            # pick K sample points including endpoints and interior
            K = 7
            xs = range(a, b; length=K)

            @inbounds for x in xs
                v_direct = ϕ(x)                     # BSplineKit evaluation
                v_poly   = poly_eval_on_piece(ϕ, piece, x)    # centered-monomial expansion
                @test isapprox(v_direct, v_poly; rtol=rtol, atol=atol)
            end

            # Also probe a couple of random interior points strictly inside (a,b)
            if b - a > 10*eps(T)
                for _ in 1:3
                    x = a + (b - a)*rand(T)  # uniform in (a,b)
                    v_direct = ϕ(x)
                    v_poly   = poly_eval_on_piece(ϕ, piece, x)
                    @test isapprox(v_direct, v_poly; rtol=rtol, atol=atol)
                end
            end
        end
    end
end


######################################################
# Test for l1_norm_difference vs QuadGK
######################################################

@testset "l1_norm_difference vs QuadGK reference ($T)" for T in [Float64, Double64]
    ALPHA  = T(0.7)
    ε      = T(1e-6)
    DEGREE = 3
    N_BP   = 30

    zero_atol = T === Float64 ? T(1e-14) : T(1e-25)
    quad_rtol = T === Float64 ? T(1e-10) : T(1e-14)
    # The QuadGK reference uses find_zeros to locate sign changes, while
    # l1_norm_difference uses a more robust approach.
    # Slight kink-location disagreement between the two methods introduces relative
    # error in the integral, so we use a looser tolerance.
    test_rtol = T === Float64 ? T(1e-6)  : T(1e-16)

    break_points = hybrid_equidistribution_breakpoints(ALPHA, ε, N_BP, T)
    B    = BSplines.BSplineBasis(BSplineOrder(DEGREE + 1), copy(break_points))
    spls = build_single_splines(B)
    n    = length(spls)

    # Unique breakpoints in [0,1] used as QuadGK subinterval boundaries
    τ_ref = sort!(unique!(filter(x -> zero(T) ≤ x ≤ one(T), collect(T, BSplineKit.knots(B)))))

    # QuadGK reference: ∫₀¹ |h(x)| dx, integrating piecewise to avoid kinks.
    function quadgk_l1(c1, c2)
            h = BSplineKit.Splines.Spline(B, c1 .- c2)
            total = zero(T)

            for k in 1:length(τ_ref)-1
                a, b = τ_ref[k], τ_ref[k+1]
                b - a < eps(T) && continue

                mid1 = (a + b) / 2
                mid2 = .25a + .75b
                mid3 = .75a + .25b

                if isapprox(h(a),    zero(T), atol=zero_atol) &&
                   isapprox(h(b),    zero(T), atol=zero_atol) &&
                   isapprox(h(mid1), zero(T), atol=zero_atol) &&
                   isapprox(h(mid2), zero(T), atol=zero_atol) &&
                   isapprox(h(mid3), zero(T), atol=zero_atol)
                    continue
                end

                internal_kinks = T[]
                try
                    internal_kinks = find_zeros(x -> h(x), a, b)
                catch e
                    if e isa DomainError
                        internal_kinks = T[]
                    else
                        rethrow(e)
                    end
                end

                breakpoints = sort!(unique!([a; internal_kinks; b]))

                val, _ = quadgk(x -> abs(h(x)), breakpoints...; rtol=quad_rtol, atol=zero(T))
                total += val
            end
            return total
        end

    Random.seed!(42)
    cases = [
        ("random dense vectors",         T.(randn(n)),  T.(randn(n))),
        ("c1 == c2 → zero norm",         T.(randn(n)),  T.(randn(n)) .* one(T)),
        ("c2 == 0 → L1 norm of spline",  T.(randn(n)),  zeros(T, n)),
        ("sparse: single basis functions", (c1=zeros(T,n); c1[n÷2]=one(T); c1),
                                           (c2=zeros(T,n); c2[n÷2+1]=one(T); c2)),
    ]
    # fix the c1==c2 case: make both vectors identical
    let v = T.(randn(n))
        cases[2] = ("c1 == c2 → zero norm", v, copy(v))
    end

    for (desc, c1, c2) in cases
            @testset "$desc" begin
                our_val = l1_norm_difference(spls, c1, c2)
                ref_val = quadgk_l1(c1, c2)
                @test isapprox(our_val, ref_val; rtol=test_rtol, atol=zero(T))
            end
        end
end


@testset "l1_norm_difference vs QuadGK reference (Mismatched Bases, $T)" for T in [Float64, Double64]
    ALPHA  = T(0.7)
    ε      = T(1e-6)
    DEGREE = 3

    zero_atol = T === Float64 ? T(1e-14) : T(1e-25)
    quad_rtol = T === Float64 ? T(1e-10) : T(1e-14)
    # Same kink-location issue as the same-basis case, compounded by the union knot mesh
    # producing more sub-intervals where find_zeros and the exact solver can disagree.
    test_rtol = T === Float64 ? T(1e-5)  : T(1e-20)

    # 1. Setup two different bases
    # Basis A: Hybrid equidistribution (Rice-like)
    N_BP_A = 30
    breaks_A = hybrid_equidistribution_breakpoints(ALPHA, ε, N_BP_A, T)

    # Basis B: Simple uniform grid
    N_BP_B = 45
    breaks_B = collect(T, range(T(0), T(1), length=N_BP_B))

    BA = BSplines.BSplineBasis(BSplineOrder(DEGREE + 1), copy(breaks_A))
    BB = BSplines.BSplineBasis(BSplineOrder(DEGREE + 1), copy(breaks_B))

    nA = length(BA)
    nB = length(BB)

    # Updated QuadGK reference to handle two different bases
    function quadgk_l1_mismatched(b1, c1, b2, c2)
        h1 = BSplineKit.Splines.Spline(b1, c1)
        h2 = BSplineKit.Splines.Spline(b2, c2)

        # We must integrate over the UNION of knots to ensure
        # h(x) is a single polynomial on every sub-interval.
        k1 = collect(T, BSplineKit.knots(b1))
        k2 = collect(T, BSplineKit.knots(b2))
        τ_union = sort!(unique!(filter(x -> zero(T) ≤ x ≤ one(T), [k1; k2])))

        total = zero(T)
        for k in 1:length(τ_union)-1
            a, b = τ_union[k], τ_union[k+1]
            b - a < 10*eps(T) && continue

            # h(x) = f(x) - g(x)
            f_diff(x) = h1(x) - h2(x)

            mid1 = (a + b) / 2
            mid2 = .25a + .75b
            mid3 = .75a + .25b
            if isapprox(f_diff(a),    zero(T), atol=zero_atol) &&
               isapprox(f_diff(b),    zero(T), atol=zero_atol) &&
               isapprox(f_diff(mid1), zero(T), atol=zero_atol) &&
               isapprox(f_diff(mid2), zero(T), atol=zero_atol) &&
               isapprox(f_diff(mid3), zero(T), atol=zero_atol)
                continue
            end

            internal_kinks = T[]
            try
                internal_kinks = find_zeros(f_diff, a, b)
            catch e
                e isa DomainError ? (internal_kinks = T[]) : rethrow(e)
            end

            breakpoints = sort!(unique!([a; internal_kinks; b]))
            val, _ = quadgk(x -> abs(f_diff(x)), breakpoints...; rtol=quad_rtol, atol=zero(T))
            total += val
        end
        return total
    end

    Random.seed!(42)

    # New cases mixing the two bases
    mismatched_cases = [
        ("Rice(30) vs Uniform(45) [Dense]",
            BA, T.(randn(nA)), BB, T.(randn(nB))),

        ("Rice(30) [f] vs Uniform(45) [g=0]",
            BA, T.(randn(nA)), BB, zeros(T, nB)),

        ("Sparse A vs Sparse B",
            BA, (v=zeros(T,nA); v[nA÷2]=one(T); v),
            BB, (v=zeros(T,nB); v[nB÷2]=one(T); v)),

        ("Large Coefficients",
            BA, T.(randn(nA) .* 1e5),
            BB, T.(randn(nB) .* 1e5))
    ]

    for (desc, b1, c1, b2, c2) in mismatched_cases
        @testset "$desc" begin
            our_val = l1_norm_difference(b1, c1, b2, c2)
            ref_val = quadgk_l1_mismatched(b1, c1, b2, c2)
            @test isapprox(our_val, ref_val; rtol=test_rtol, atol=zero(T))
        end
    end
end


######################################################
# Tests for hybrid_equidistribution_breakpoints
######################################################
#
# The mesh contract (see docstring and the paper's "Optimal Knot Placement"
# section):
#   * breakpoints on [0,1], strictly increasing, symmetric about 1/2;
#   * contains 0, L_ε (boundary-layer scale), 1/2, and 1;
#   * outer region [L_ε, 1/2]: de Boor power partition with γ = (1-2α)/9,
#     degenerating to log-linear spacing at α = 1/2;
#   * inner region [0, L_ε]: geometric grading with ratio r ∈ [1,2], matched to
#     the first outer step h₀ at L_ε; no interior inner points when h₀ ≥ L_ε.
#
# HB-1 covers the structural invariants; HB-2..HB-4 pin the closed forms of the
# two outer branches and the inner grading; HB-5 is argument validation.

function check_hbp_invariants(pts, α, ε; T=Float64)
    ξ = boundary_layer_scale(T(α), T(ε))

    @test pts[1]   == zero(T)
    @test pts[end] == one(T)
    @test all(zero(T) .≤ pts .≤ one(T))
    @test issorted(pts)
    @test length(pts) == length(unique(pts))

    n = length(pts)
    for i in 1:n
        @test pts[i] + pts[n + 1 - i] ≈ one(T) atol=4*eps(T) # symmetric about 1/2
    end

    @test any(p -> abs(p - ξ) ≤ 4*eps(T)*ξ, pts) # at least one breakpoint within 4*eps(ξ) of ξ
end

@testset "hybrid_equidistribution_breakpoints: structural invariants (HB-1)" begin
    # Float64 sweep over a wide range of (α, ε, n_outer), including the coarse
    # and large-ε corners.
    T = Float64
    for (α, ε, n_outer) in [(0.3, 1e-4, 150), (0.5, 1e-4, 100), (0.7, 1e-4, 50),
                             (0.3, 1e-2, 5),  (0.9, 1e-6, 20)]
        pts = hybrid_equidistribution_breakpoints(α, ε, n_outer, T)
        check_hbp_invariants(pts, α, ε; T=T)
    end

    # Both precisions, with the stronger claim that the mandatory points are
    # placed *exactly* rather than merely approximated, and that the mesh is
    # strictly increasing.
    for T in (Float64, Double64),
        α in (0.25, 0.5, 0.75),
        ε in (1e-3, 1e-6, 1e-10)

        αT, εT  = T(α), T(ε)
        n_outer = 16
        b = hybrid_equidistribution_breakpoints(αT, εT, n_outer, T)
        L = boundary_layer_scale(αT, εT)

        @test b[1] == zero(T)
        @test b[end] == one(T)
        @test all(diff(b) .> zero(T))                    # strictly increasing

        # symmetric about 1/2 (reflection is computed as 1 - x, so allow 2 ulps)
        n = length(b)
        @test all(abs(b[i] + b[n+1-i] - one(T)) ≤ 2eps(T) for i in 1:n)

        # mandatory points present exactly
        @test any(x -> x == L, b)
        @test any(x -> x == one(T)/2, b)
    end
end

@testset "hybrid_equidistribution_breakpoints: outer power partition closed form (HB-2)" begin
    for T in (Float64, Double64), α in (0.25, 0.75)
        αT, εT  = T(α), T(1e-6)
        n_outer = 12
        b  = hybrid_equidistribution_breakpoints(αT, εT, n_outer, T)
        L  = boundary_layer_scale(αT, εT)
        γ  = (one(T) - 2αT) / T(9)

        iL = findfirst(==(L), b)
        @test iL !== nothing
        outer = b[iL+1 : iL+n_outer-1]                   # interior outer points
        @test b[iL + n_outer] == one(T)/2                # next point after them is 1/2

        Lγ = L^γ
        Δγ = (one(T)/2)^γ - Lγ
        for (i, x) in enumerate(outer)
            x_ref = (Lγ + T(i)/T(n_outer) * Δγ)^(one(T)/γ)
            @test isapprox(x, x_ref; rtol=8eps(T))
        end
    end
end

@testset "hybrid_equidistribution_breakpoints: α = 1/2 log-linear branch (HB-3)" begin
    # γ = (1-2α)/9 vanishes at α = 1/2, so the power partition degenerates to
    # log-linear spacing.  Structure first, then the closed form.
    T = Float64
    for n_outer in [2, 5, 15]
        pts = hybrid_equidistribution_breakpoints(0.5, 1e-4, n_outer, T)
        check_hbp_invariants(pts, 0.5, 1e-4; T=T)
    end

    αT, εT  = T(0.5), T(1e-6)
    n_outer = 12
    b = hybrid_equidistribution_breakpoints(αT, εT, n_outer, T)
    L = boundary_layer_scale(αT, εT)

    iL    = findfirst(==(L), b)
    outer = b[iL+1 : iL+n_outer-1]
    logL, logh = log(L), log(T(0.5))
    for (i, x) in enumerate(outer)
        x_ref = exp(logL + T(i)/T(n_outer) * (logh - logL))
        @test isapprox(x, x_ref; rtol=8eps(T))
    end
end

@testset "hybrid_equidistribution_breakpoints: inner geometric grading (HB-4)" begin
    for T in (Float64, Double64), α in (0.25, 0.5, 0.75), ε in (1e-3, 1e-6)
        αT, εT  = T(α), T(ε)
        n_outer = 16
        b  = hybrid_equidistribution_breakpoints(αT, εT, n_outer, T)
        L  = boundary_layer_scale(αT, εT)
        iL = findfirst(==(L), b)
        h0 = b[iL+1] - L                                 # first outer step

        inner_diffs = diff(b[1:iL])                      # steps of [0, ..., L]
        tol = sqrt(eps(T))

        if h0 ≥ L
            # Degenerate case: a single interval [0, L] suffices
            @test iL == 2
        else
            # Matching condition: the step adjacent to L is no larger than h₀
            @test inner_diffs[end] ≤ h0 * (1 + tol)

            # Grading: steps grow leftward (away from L) by a constant ratio
            # r ∈ [1, 2] — except the leftmost interval, which in the uniform
            # case (r = 1) is the remainder L - (K-1)h₀ and may be shorter
            # than its neighbour. So every ratio is ≤ 2, and all ratios past
            # the leftmost pair are ≥ 1.
            if length(inner_diffs) ≥ 2
                ratios = inner_diffs[1:end-1] ./ inner_diffs[2:end]
                @test all(ratios .≤ 2 + tol)
                @test all(ratios[2:end] .≥ 1 - tol)
            end
        end
    end

    # The degenerate h₀ ≥ L_ε regime reached through a large ε, checked for the
    # full structural contract rather than just the interior-point count.
    pts = hybrid_equidistribution_breakpoints(0.5, 0.3, 5, Float64)
    check_hbp_invariants(pts, 0.5, 0.3; T=Float64)
end

@testset "hybrid_equidistribution_breakpoints: argument validation (HB-5)" begin
    @test_throws ArgumentError hybrid_equidistribution_breakpoints(0.0,  1e-4, 10)
    @test_throws ArgumentError hybrid_equidistribution_breakpoints(1.0,  1e-4, 10)
    @test_throws ArgumentError hybrid_equidistribution_breakpoints(-0.1, 1e-4, 10)
    @test_throws ArgumentError hybrid_equidistribution_breakpoints(0.5,  0.0,  10)
    @test_throws ArgumentError hybrid_equidistribution_breakpoints(0.5, -1e-4, 10)
    @test_throws ArgumentError hybrid_equidistribution_breakpoints(0.5,  1e-4,  0)
end


######################################################
# merge_breakpoints
######################################################

@testset "merge_breakpoints" begin
    T = Float64

    # Basic merge of two disjoint sorted arrays
    pts1 = T[0.0, 0.25, 0.5]
    pts2 = T[0.1, 0.3, 0.75, 1.0]
    result = merge_breakpoints(pts1, pts2)
    @test result == T[0.0, 0.1, 0.25, 0.3, 0.5, 0.75, 1.0]

    # Duplicates within atol are collapsed
    pts3 = T[0.0, 0.5, 1.0]
    pts4 = T[0.0 + 1e-16, 0.5, 1.0]
    r2 = merge_breakpoints(pts3, pts4; atol=T(1e-14))
    @test r2 == T[0.0, 0.5, 1.0]

    # One empty array
    @test merge_breakpoints(T[], T[0.2, 0.8]) == T[0.2, 0.8]
    @test merge_breakpoints(T[0.2, 0.8], T[]) == T[0.2, 0.8]

    # Both empty
    @test merge_breakpoints(T[], T[]) == T[]

    # Identical arrays — output should be deduplicated
    pts5 = T[0.0, 0.3, 0.7, 1.0]
    @test merge_breakpoints(pts5, pts5; atol=T(1e-14)) == pts5
end


######################################################
# convert_precision  (SingleBSpline round-trip)
######################################################

@testset "convert_precision round-trip Float64 to Double64 and back" begin
    α, ε = 0.5, 0.1
    basis64, _ = build_pm_basis(α, ε, 10, Float64)
    spls64 = build_single_splines(basis64)

    # Two evaluation points per knot span so every span is covered
    bps = unique!(sort!(collect(Float64, BSplineKit.knots(basis64))))
    xs = Float64[]
    for k in 1:length(bps)-1
        a, b = bps[k], bps[k+1]
        push!(xs, .25a + .75b, .75a + .25b)
    end

    for ϕ in spls64
        ϕ128 = convert_precision(ϕ, Double64)
        @test ϕ128 isa PmSpectrum.Bases.SingleBSpline{Double64}
        for x in xs
            @test Float64(ϕ128(Double64(x))) ≈ ϕ(x)  rtol=1e-12  atol=0.0
        end
    end

    # Vector overload: convert all at once
    spls128 = convert_precision(spls64, Double64)
    @test length(spls128) == length(spls64)
    @test all(s -> s isa PmSpectrum.Bases.SingleBSpline{Double64}, spls128)

    # No-op when T === S
    spls_same = convert_precision(spls64, Float64)
    @test spls_same === spls64   # same object returned, no copy
end


######################################################
# _cubic_roots_in and _centpoly_roots_in
######################################################

@testset "_cubic_roots_in" begin
    cri = PmSpectrum.Bases._cubic_roots_in
    atol = 1e-12

    # Three sign-changing roots, all in the interval.
    # t*(t-0.2)*(t+0.2) = t³ - 0.04t.
    #
    # The routine may also return the two critical points as conservative
    # integration breakpoints.
    roots = cri(0.0, -0.04, 0.0, 1.0, -0.3, 0.3, atol)

    for r in (-0.2, 0.0, 0.2)
        @test any(x -> isapprox(x, r; atol=1e-12), roots)
    end

    # The critical points ±sqrt(0.04/3) are retained as conservative breakpoints.
    rcrit = sqrt(0.04 / 3)
    @test any(x -> isapprox(x, -rcrit; atol=1e-12), roots)
    @test any(x -> isapprox(x,  rcrit; atol=1e-12), roots)

    # Three sign-changing roots with nonzero quadratic coefficient.
    # (t-0.5)*(t-0.1)*(t+0.3) = t³ - 0.3t² - 0.13t + 0.015.
    roots = cri(0.015, -0.13, -0.3, 1.0, -0.4, 0.6, atol)

    for r in (-0.3, 0.1, 0.5)
        @test any(x -> isapprox(x, r; atol=1e-12), roots)
    end

    # Interval filter: the only sign-changing root in (0.05, 0.3) is 0.2.
    # Additional critical-point breakpoints are allowed.
    roots = cri(0.0, -0.04, 0.0, 1.0, 0.05, 0.3, atol)

    @test any(x -> isapprox(x, 0.2; atol=1e-12), roots)
    @test all(x -> 0.05 + atol < x < 0.3 - atol, roots)

    # Monotone cubic with one real root: t³ + t + 1.
    # Since p'(t) = 3t² + 1 has no real roots, no extra critical-point
    # breakpoints are introduced.
    roots = cri(1.0, 1.0, 0.0, 1.0, -1.0, 0.5, atol)

    @test length(roots) == 1
    @test roots[1]^3 + roots[1] + 1 ≈ 0.0 atol=1e-12

    # Triple root at zero: t³.
    # The derivative has a repeated root at zero, which is retained as a
    # conservative breakpoint.
    roots = cri(0.0, 0.0, 0.0, 1.0, -0.5, 0.5, atol)

    @test roots ≈ [0.0] atol=1e-12

    # Roots at ±0.2 lie on the boundaries and are excluded. The interior root
    # at zero must still be found, while the two interior critical points may
    # also be returned.
    roots = cri(0.0, -0.04, 0.0, 1.0, -0.2, 0.2, 1e-10)

    @test any(x -> isapprox(x, 0.0; atol=1e-12), roots)
    @test all(x -> -0.2 + 1e-10 < x < 0.2 - 1e-10, roots)

    # All roots and critical points lie outside the interval.
    # (t-1)*(t-2)*(t-3) = t³ - 6t² + 11t - 6.
    roots = cri(-6.0, 11.0, -6.0, 1.0, -0.5, 0.5, atol)

    @test roots == Float64[]
end


@testset "_centpoly_roots_in" begin
    cpri = PmSpectrum.Bases._centpoly_roots_in
    atol = 1e-12

    # Degree 0 (constant): no breakpoints.
    @test cpri([5.0], -1.0, 1.0, atol) == Float64[]

    # Degree 1 (linear): c[1] + c[2]*t = 0  →  t = 0.1.
    roots = cpri([-0.1, 1.0], -0.5, 0.5, atol)
    @test roots ≈ [0.1] atol=1e-12

    # Degree 2 with two real roots: t² - 0.04 = 0  →  t = ±0.2.
    roots = cpri([-0.04, 0.0, 1.0], -0.3, 0.3, atol)
    @test roots ≈ [-0.2, 0.2] atol=1e-12

    # Degree 2 with a repeated root.
    # t² = 0 → conservative breakpoint at zero.
    roots = cpri([0.0, 0.0, 1.0], -1.0, 1.0, atol)
    @test roots ≈ [0.0] atol=1e-12

    # Degree 2 with no real roots: t² + 1 = 0.
    @test cpri([1.0, 0.0, 1.0], -1.0, 1.0, atol) == Float64[]

    # Degree 3 delegates to _cubic_roots_in.
    # t³ - 0.04t has sign-changing roots at -0.2, 0, 0.2.
    # The cubic routine may additionally return critical points.
    roots = cpri([0.0, -0.04, 0.0, 1.0], -0.3, 0.3, atol)

    for r in (-0.2, 0.0, 0.2)
        @test any(x -> isapprox(x, r; atol=1e-12), roots)
    end

    # All returned cubic breakpoints lie strictly inside the requested span.
    @test all(x -> -0.3 + atol < x < 0.3 - atol, roots)

    # Trailing-zero stripping: length-4 vector with effective degree 1.
    # c = [-0.1, 1.0, 0, 0] should be treated identically to [-0.1, 1.0].
    roots = cpri([-0.1, 1.0, 0.0, 0.0], -0.5, 0.5, 1e-10)
    @test roots ≈ [0.1] atol=1e-12
end


