# --------------------------
# Test for Utils.jl
# --------------------------


using Test
using Random: MersenneTwister
using LinearAlgebra: Symmetric, I, norm, eigvals
using GaussQuadrature: legendre
using PmSpectrum.Utils: log10_fit_lsq,
    symmetric_pm, symmetric_pm_derivative,
    scaled_nonsymmetric_eigen,
    _gl_integrate, _gl_integrate_mesh,
    match_nearest, track_eigenvalues, order_disagreements


######################################################
# The symmetric PM map
######################################################

@testset "symmetric_pm: branch formulas and the x = 1/2 convention" begin
    for α in (0.15, 0.5, 0.85)
        left(x)  = x + 2.0^α * x^(1 + α)
        right(x) = x - 2.0^α * (1 - x)^(1 + α)

        @test symmetric_pm(0.0, α) == 0.0
        @test symmetric_pm(1.0, α) == 1.0

        for x in (1e-8, 0.01, 0.2, 0.4, 0.49)
            @test symmetric_pm(x, α) ≈ left(x)  rtol=1e-14
        end
        for x in (0.51, 0.6, 0.8, 0.99, 1 - 1e-8)
            @test symmetric_pm(x, α) ≈ right(x) rtol=1e-14
        end

        # x = 1/2 belongs to the LEFT branch, so T(1/2) = 1 — not the 0 the
        # right-branch formula would give. Same point on the circle, different
        # representative on [0,1]; `Branch` metadata reads the representative.
        # Within a couple of ulp, not exact: 2^α and (1/2)^(1+α) are each rounded
        # before multiplying, so their exact cancellation to 1/2 is not recovered.
        @test symmetric_pm(0.5, α) ≈ 1.0 atol=4eps(Float64)
        @test right(0.5)           ≈ 0.0 atol=4eps(Float64)

        # T(1-x) = 1 - T(x) away from the shared breakpoint
        for x in (0.01, 0.2, 0.4, 0.49)
            @test symmetric_pm(1 - x, α) ≈ 1 - symmetric_pm(x, α) rtol=1e-14
        end

        # derivative is the same expression in d = min(x, 1-x) on both branches
        @test symmetric_pm_derivative(0.3, α) ≈ symmetric_pm_derivative(0.7, α) rtol=1e-14
        @test symmetric_pm_derivative(0.3, α) ≥ 1
    end

    @test_throws ArgumentError symmetric_pm(-0.1, 0.5)
    @test_throws ArgumentError symmetric_pm(1.1, 0.5)
end


######################################################
# Exact power law — slope and intercept recovered
######################################################

@testset "log10_fit_lsq: exact power law y = x^m" begin
    xs = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    # y = x^2: slope 2, intercept 0
    m, a10 = log10_fit_lsq(xs, xs .^ 2)
    @test m   ≈ 2.0 rtol=1e-12
    @test a10 ≈ 0.0 atol=1e-12

    # y = 3 * x^2: slope 2, intercept log10(3)
    m, a10 = log10_fit_lsq(xs, 3.0 .* xs .^ 2)
    @test m   ≈ 2.0       rtol=1e-12
    @test a10 ≈ log10(3.0) rtol=1e-12
end


######################################################
# Negative exponent recovered
######################################################

@testset "log10_fit_lsq: negative exponent y = x^(-0.5)" begin
    xs = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    m, a10 = log10_fit_lsq(xs, xs .^ (-0.5))
    @test m   ≈ -0.5 rtol=1e-12
    @test a10 ≈  0.0 atol=1e-12
end


######################################################
# Non-finite / non-positive values are filtered
######################################################

@testset "log10_fit_lsq: bad values filtered without changing the fit" begin
    xs_clean = [0.1, 0.5, 1.0, 2.0, 5.0]
    ys_clean = xs_clean .^ 1.5

    m_ref, a10_ref = log10_fit_lsq(xs_clean, ys_clean)

    # Inject bad entries: x ≤ 0, y ≤ 0, Inf, NaN
    xs_bad = vcat(-1.0, 0.0, xs_clean, Inf,  1.0)
    ys_bad = vcat( 1.0, 1.0, ys_clean,  1.0, -1.0)

    m, a10 = @test_logs (:warn,) log10_fit_lsq(xs_bad, ys_bad)
    @test m   ≈ m_ref   rtol=1e-12
    @test a10 ≈ a10_ref rtol=1e-12
end


@testset "log10_fit_lsq: underdetermined fits are rejected" begin
    @test_throws ArgumentError log10_fit_lsq([1.0, 2.0], [1.0])       # length mismatch
    @test_throws ArgumentError log10_fit_lsq([-1.0, 0.0], [1.0, 2.0]) # nothing survives

    # One surviving pair: the 1×2 system is underdetermined and a least-squares
    # solve would happily return a minimum-norm "slope" fitted to a single point.
    @test_throws ArgumentError log10_fit_lsq([1.0], [2.0])
    @test_throws ArgumentError log10_fit_lsq([2.0, -1.0], [3.0, 1.0])

    # Two points but one distinct x: equally rank-deficient in the slope column.
    @test_throws ArgumentError log10_fit_lsq([2.0, 2.0], [1.0, 3.0])
    @test_throws ArgumentError log10_fit_lsq([2.0, 2.0, 2.0], [1.0, 3.0, 9.0])

    # Two distinct x-values is the minimum that does fit: exact through 2 points.
    m, a10 = log10_fit_lsq([1.0, 10.0], [2.0, 20.0])
    @test m   ≈ 1.0        rtol=1e-12
    @test a10 ≈ log10(2.0) rtol=1e-12
end


######################################################
# scaled_nonsymmetric_eigen
######################################################

@testset "scaled_nonsymmetric_eigen eigenvalues match base eigvals" begin
    rng = MersenneTwister(42)
    n = 6; T = Float64
    A = randn(rng, T, n, n)
    M = Symmetric(A'A + T(n) * I)   # well-conditioned SPD
    G = randn(rng, T, n, n)

    vals, _ = scaled_nonsymmetric_eigen(G, M)
    vals_ref = eigvals(G, M)

    perm   = sortperm(real.(vals))
    perm_r = sortperm(real.(vals_ref))
    @test real.(vals[perm])  ≈ real.(vals_ref[perm_r])  rtol=1e-10 atol=0.0
    @test imag.(vals[perm])  ≈ imag.(vals_ref[perm_r])  rtol=1e-10 atol=1e-12
end

@testset "scaled_nonsymmetric_eigen eigenvectors satisfy G v = λ M v" begin
    rng = MersenneTwister(42)
    n = 6; T = Float64
    A = randn(rng, T, n, n)
    M = Symmetric(A'A + T(n) * I)
    G = randn(rng, T, n, n)

    vals, vecs = scaled_nonsymmetric_eigen(G, M)
    for k in 1:n
        v = vecs[:, k]; λ = vals[k]
        @test norm(G * v - λ * (M * v)) ≤ 1e-10 * (norm(M * v) * abs(λ) + 1.0)
    end
end

@testset "scaled_nonsymmetric_eigen throws ArgumentError for non-symmetric M" begin
    rng = MersenneTwister(7)
    n = 4
    M_nonsym = rand(rng, n, n)   # generically not symmetric
    G = rand(rng, n, n)
    @test_throws ArgumentError scaled_nonsymmetric_eigen(G, M_nonsym)
end

@testset "scaled_nonsymmetric_eigen throws ArgumentError for indefinite M" begin
    # Positive diagonal but indefinite: top-left 2×2 block [2 3; 3 2] has det < 0.
    # sqrt.(diag(M)) succeeds; Cholesky catches the indefiniteness.
    M_indef = Symmetric([2.0 3.0 0.0 0.0;
                          3.0 2.0 0.0 0.0;
                          0.0 0.0 1.0 0.0;
                          0.0 0.0 0.0 1.0])
    G = rand(MersenneTwister(7), 4, 4)
    @test_throws ArgumentError scaled_nonsymmetric_eigen(G, M_indef)
end


######################################################
# _gl_integrate
######################################################

@testset "_gl_integrate: exact for polynomial of degree ≤ 2p-1" begin
    T = Float64
    p = 5   # 5-point GL is exact for degree ≤ 9
    ξ, ω = legendre(T, p)
    a, b = T(0.3), T(1.7)

    f = x -> x^9
    val = _gl_integrate(f, a, b, ξ, ω)
    ref = (b^10 - a^10) / T(10)
    @test val ≈ ref rtol=1e-12 atol=0.0
end

@testset "_gl_integrate: matches QuadGK for smooth function" begin
    T = Float64
    p = 8
    ξ, ω = legendre(T, p)
    a, b = T(0.1), T(0.9)

    f = x -> exp(-x) * sin(3x)
    val = _gl_integrate(f, a, b, ξ, ω)
    ref, _ = quadgk(f, a, b; rtol=1e-14)
    @test val ≈ ref rtol=1e-12 atol=0.0
end


######################################################
# _gl_integrate_mesh
######################################################

@testset "_gl_integrate_mesh: matches QuadGK over piecewise-smooth mesh" begin
    T = Float64
    p = 6
    ξ, ω = legendre(T, p)
    breaks = T[0.0, 0.25, 0.5, 0.75, 1.0]

    f = x -> exp(-x) * cos(x)
    val = _gl_integrate_mesh(f, breaks, ξ, ω)
    ref, _ = quadgk(f, T(0), T(1); rtol=1e-14)
    @test val ≈ ref rtol=1e-12 atol=0.0
end

@testset "_gl_integrate_mesh: sub-interval narrower than atol is skipped" begin
    T = Float64
    p = 4
    ξ, ω = legendre(T, p)
    gap = T(1e-10)
    
    # Near-duplicate breakpoint — the tiny sub-interval [0.5, 0.5+gap] is skipped.
    # The two remaining intervals integrate f=1 to exactly 1 - gap.
    breaks = T[0.0, 0.5, 0.5 + gap, 1.0]
    val = _gl_integrate_mesh(x -> one(T), breaks, ξ, ω; atol=T(1e-6))
    @test val ≈ one(T) - gap  rtol=1e-12 atol=0.0
end


######################################################
# Eigenvalue tracking utilities
######################################################

@testset "match_nearest: identity assignment on equal spectra" begin
    z = Complex{Float64}[1.0 + 0.0im, 0.7 + 0.2im, 0.7 - 0.2im, 0.3 + 0.0im]
    @test match_nearest(z, z) == collect(1:length(z))
end

@testset "match_nearest: full complex value (no conjugate mix-up)" begin
    # ref = the +imag member of a conjugate pair plus a real eigenvalue.
    ref  = Complex{Float64}[0.6 + 0.3im, 0.9 + 0.0im]
    # curr holds both conjugate partners and the real one, in scrambled order.
    curr = Complex{Float64}[0.9 + 0.0im, 0.6 - 0.3im, 0.6 + 0.3im]
    idx  = match_nearest(ref, curr)
    @test curr[idx[1]] == 0.6 + 0.3im   # matched to +imag partner, not 0.6 - 0.3im
    @test curr[idx[2]] == 0.9 + 0.0im
end

@testset "track_eigenvalues: no perturbation ⇒ order preserved" begin
    s = Complex{Float64}[1.0 + 0.0im, 0.8 + 0.1im, 0.8 - 0.1im, 0.4 + 0.0im]
    spectra = [copy(s) for _ in 1:4]
    tracked, idxmap = track_eigenvalues(spectra; k_track = 3)

    @test size(tracked) == (3, 4)
    @test size(idxmap)  == (3, 4)
    @test all(idxmap .== [k for k in 1:3, _ in 1:4])   # every mode stays at its rank
    @test isempty(order_disagreements(idxmap))
    for ℓ in 1:4, k in 1:3
        @test tracked[k, ℓ] == s[k]
    end
end

@testset "track_eigenvalues: small jitter keeps naive labeling" begin
    base = Complex{Float64}[1.0 + 0.0im, 0.70 + 0.0im, 0.40 + 0.0im, 0.10 + 0.0im]
    # Perturb each by ≪ the 0.3 inter-eigenvalue spacing; order cannot change.
    jit  = [Complex{Float64}[b + 1e-3 * (ℓ) for b in base] for ℓ in 0:3]
    tracked, idxmap = track_eigenvalues(jit; k_track = 4)
    @test isempty(order_disagreements(idxmap))
    for ℓ in 1:4, k in 1:4
        @test tracked[k, ℓ] == jit[ℓ][k]   # tracked value == naive k-th entry
    end
end

@testset "track_eigenvalues: magnitude crossing is followed and flagged" begin
    # Two modes A and B, each stored |λ|-descending within its level.
    # Level 1 (reference, most accurate): A=0.90 (rank 1), B=0.80+0.30im (rank 2).
    # Level 2: A drifts down to 0.82, B drifts up to 0.86+0.30im ⇒ |B| > |A|, so the
    # stored (magnitude-sorted) order swaps: B is now rank 1, A rank 2.
    A1, B1 = 0.90 + 0.00im, 0.80 + 0.30im     # |A1|=0.90, |B1|≈0.854
    A2, B2 = 0.82 + 0.00im, 0.86 + 0.30im     # |A2|=0.82, |B2|≈0.911
    lvl1 = sort(Complex{Float64}[A1, B1]; by = abs, rev = true)   # [A1, B1]
    lvl2 = sort(Complex{Float64}[A2, B2]; by = abs, rev = true)   # [B2, A2]

    tracked, idxmap = track_eigenvalues([lvl1, lvl2]; k_track = 2)

    # Mode 1 (=A) is followed continuously despite dropping to magnitude rank 2.
    @test tracked[1, 1] == A1
    @test tracked[1, 2] == A2
    @test tracked[2, 1] == B1
    @test tracked[2, 2] == B2
    @test abs(tracked[1, 2] - tracked[1, 1]) < 0.1   # small, continuous step
    @test abs(tracked[2, 2] - tracked[2, 1]) < 0.1

    # idxmap records the swap at level 2, and both modes are flagged.
    @test idxmap[:, 1] == [1, 2]
    @test idxmap[:, 2] == [2, 1]
    @test Set(order_disagreements(idxmap)) == Set([(1, 2), (2, 2)])
end

@testset "track_eigenvalues: truncation to k_track leading modes" begin
    s = Complex{Float64}[1.0 + 0.0im, 0.6 + 0.1im, 0.6 - 0.1im, 0.3 + 0.0im, 0.05 + 0.0im]
    spectra = [copy(s), copy(s)]
    tracked, idxmap = track_eigenvalues(spectra; k_track = 2)
    @test size(tracked) == (2, 2)
    @test size(idxmap)  == (2, 2)
    @test tracked[:, 1] == s[1:2]
end

@testset "tracking utilities: input validation" begin
    z = Complex{Float64}[1.0 + 0.0im, 0.5 + 0.0im]
    @test_throws DimensionMismatch match_nearest(z, z[1:1])          # curr too short
    @test_throws DimensionMismatch track_eigenvalues([z, z[1:1]]; k_track = 2)
    @test_throws ArgumentError track_eigenvalues([z]; k_track = 0)
end
