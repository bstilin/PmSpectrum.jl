# ---------------------------------------------------
# Tests for ResolventNorm.jl
# ---------------------------------------------------
#
# reduced_resolvent_norm computes
#
#     M_{ε,N} = ‖ [ (I - P_{ε,N})|_{V_{N,0}} ]⁻¹ ‖_{L²→L²} = 1 / σ_min(Q₀* B Q₀)
#
# from the Galerkin pair (M, G) and the exact integration functional m, working
# in the L²-orthonormal coordinates y = L* D⁻¹ c.  What is tested:
#
#   RN-1  basis-scaling invariance: the diagonal equilibration is only a change
#         of basis, so the unscaled calculation must give the same answer
#   RN-2  the zero-mass constraint: Q₀* q = 0 and Q₀* Q₀ = I
#   RN-2b argument validation: q ≠ 0, and M's definiteness caught before the
#         equilibration rather than after it
#   RN-3  mass preservation of the discretisation: q* B = 0
#   RN-4  agreement with an independent route through the explicit A = M⁻¹G
#   RN-5  a normal operator, where the answer is known: 1 / min_{λ≠1} |1 - λ|
#
# ε is deliberately large (0.05) so the basis stays well conditioned enough that
# the unscaled reference calculations of RN-1 and RN-4 are meaningful; the whole
# point of the equilibration is that they stop being so as ε → 0.
#
# Double64 tolerances are set near 1e-16, not near eps(Double64): the assembled
# transfer matrix satisfies its mass identity only to ~2e-17, a floor in the
# assembly path rather than round-off.

using Test
using PmSpectrum
using PmSpectrum.Bases: build_single_splines
using PmSpectrum.ResolventNorm: orthonormal_complement, l2_orthonormal_form,
    reduced_resolvent_norm, mass_tolerance
using LinearAlgebra: cholesky, Symmetric, Diagonal, normalize, norm, svdvals, adjoint, I


# One solve per scalar type, shared by every set below.  N = 40 breakpoints
# (n = 93 basis functions) keeps each one to a couple of seconds.
const RN_ALPHA = 0.5
const RN_EPS   = 0.05
const RN_NBP   = 40
const RN_NQ    = 32

function rn_setup(::Type{T}) where {T}
    r    = run_single_experiment(T(RN_ALPHA), T(RN_EPS), RN_NBP, RN_NQ)
    spls = build_single_splines(rehydrate(r).basis)
    m    = T[ϕ.mass for ϕ in spls]
    return (M = r.M, G = r.G, m = m, out = reduced_resolvent_norm(r.M, r.G, m))
end

const RN = Dict{DataType,Any}(T => rn_setup(T) for T in (Float64, Double64))


rn_rtol(::Type{Float64})  = 1e-12
rn_rtol(::Type{Double64}) = Double64(1e-28)

rn_mass_bound(::Type{Float64})  = 1e-13
rn_mass_bound(::Type{Double64}) = Double64(1e-15)


@testset "RN-1 basis-scaling invariance ($T)" for T in (Float64, Double64)
    s = RN[T]

    # The same calculation with D = I: M = L L*, B = L⁻¹ (M - G) L⁻*, q = L⁻¹ m.
    # Equilibration is a change of basis and must not move the answer.
    L  = cholesky(Symmetric(s.M)).L
    B  = (L \ (s.M - s.G)) / adjoint(L)
    q  = normalize(L \ s.m)
    Q0 = orthonormal_complement(q)
    σ  = minimum(svdvals(adjoint(Q0) * B * Q0))

    @test σ ≈ s.out.sigma_min rtol=rn_rtol(T)
    @test inv(σ) ≈ s.out.norm rtol=rn_rtol(T)

    # The equilibration is worth doing even at this comfortable ε.
    @test s.out.cond_mass_scaled < s.out.cond_mass
end


@testset "RN-2 zero-mass constraint ($T)" for T in (Float64, Double64)
    # A short vector first, where the tolerance can be a small multiple of eps.
    q  = normalize(T[1, 2, 3, 4, 5, 6, 7])
    Q0 = orthonormal_complement(q)
    @test size(Q0) == (7, 6)
    @test norm(adjoint(Q0) * q) ≤ 10 * eps(T)
    @test maximum(abs, adjoint(Q0) * Q0 - I) ≤ 10 * eps(T)

    # Then the q that the calculation actually uses.
    s  = RN[T]
    f  = l2_orthonormal_form(s.M, s.G, s.m)
    Q0 = orthonormal_complement(f.q)
    n  = length(f.q)
    @test size(Q0) == (n, n - 1)
    @test norm(adjoint(Q0) * f.q) ≤ 100 * eps(T)
    @test maximum(abs, adjoint(Q0) * Q0 - I) ≤ 100 * eps(T)

    @test_throws ArgumentError orthonormal_complement(T[1])

    # q = 0 has no (n-1)-dimensional complement — its complement is all of Rⁿ —
    # but the QR is perfectly happy to hand back an arbitrary orthonormal block,
    # silently satisfying `Q₀* q = 0` for the wrong reason.
    @test_throws ArgumentError orthonormal_complement(zeros(T, 5))
end


@testset "RN-2b l2_orthonormal_form rejects non-SPD M ($T)" for T in (Float64, Double64)
    G = Matrix{T}(I, 2, 2)
    m = T[1, 1]

    @test_throws ArgumentError l2_orthonormal_form(T[1 0; 0 1], G, T[1])      # length
    @test_throws ArgumentError l2_orthonormal_form(T[1 2; 3 1], G, m)         # asymmetric

    # A nonpositive diagonal entry has to be caught BEFORE D = diag(M)^(-1/2) is
    # formed: sqrt of it gives NaN/Inf, and the Cholesky on a NaN-laden matrix
    # would not reliably raise the PosDefException that the clean error rides on.
    @test_throws ArgumentError l2_orthonormal_form(T[0 1; 1 1], G, m)         # zero diagonal
    @test_throws ArgumentError l2_orthonormal_form(T[-1 0; 0 1], G, m)        # negative diagonal

    # Positive diagonal but indefinite
    @test_throws ArgumentError l2_orthonormal_form(T[1 2; 2 1], G, m)

    # And a genuine SPD M goes through.
    f = l2_orthonormal_form(T[2 1; 1 2], G, m)
    @test size(f.B) == (2, 2)
    @test norm(f.q) ≈ one(T) rtol=10 * eps(T)
end


@testset "RN-3 mass preservation ($T)" for T in (Float64, Double64)
    s = RN[T]

    # The B-splines are a partition of unity, so M·1 = m identically and the
    # discretisation preserves mass: q* B = 0.  Neither is projected away.
    # Checked against both the production threshold (i.e. no warning fires) and
    # the sharper level actually attained.
    @test s.out.pou_error  ≤ mass_tolerance(T)
    @test s.out.mass_error ≤ mass_tolerance(T)
    @test s.out.pou_error  ≤ rn_mass_bound(T)
    @test s.out.mass_error ≤ rn_mass_bound(T)

    f = l2_orthonormal_form(s.M, s.G, s.m)
    @test norm(adjoint(f.B) * f.q) / s.out.sigma_max ≈ s.out.mass_error

    # Dimensions and argument checking.
    @test_throws ArgumentError reduced_resolvent_norm(s.M, s.G, s.m[1:end-1])
    @test_throws ArgumentError reduced_resolvent_norm(s.M, s.G[:, 1:end-1], s.m)
end


@testset "RN-4 agreement with explicit A = M⁻¹G ($T)" for T in (Float64, Double64)
    s = RN[T]

    # Independent route: form the coefficient-space operator A = M⁻¹G outright
    # (safe only because this test problem is small and well conditioned) and
    # push I - A into L²-orthonormal coordinates y = C c, where M = C* C.
    # C (I - A) C⁻¹ = C⁻* (M - G) C⁻¹, which is what the production code builds
    # from K without ever forming A.
    A    = s.M \ s.G
    C    = cholesky(Symmetric(s.M)).U
    Bind = C * (I - A) / C
    q    = normalize(adjoint(C) \ s.m)
    Q0   = orthonormal_complement(q)
    σ    = minimum(svdvals(adjoint(Q0) * Bind * Q0))

    @test σ ≈ s.out.sigma_min rtol=rn_rtol(T)
end


@testset "RN-5 normal operator, known answer ($T)" for T in (Float64, Double64)
    s = RN[T]
    n = length(s.m)

    # Build a self-adjoint, mass-preserving operator on the real basis: in
    # L²-orthonormal coordinates take P_y = q q* + Q₀ diag(λ) Q₀*, which fixes the
    # mass direction q (eigenvalue 1) and acts with the prescribed spectrum on q⊥.
    # Pulling back with G = L P_y L* gives a genuine Galerkin pair.  It is NOT a
    # Markov operator — nothing here forces positivity preservation, and λ runs
    # negative — but self-adjointness is all this test needs.
    L  = cholesky(Symmetric(s.M)).L
    q  = normalize(L \ s.m)
    Q0 = orthonormal_complement(q)
    λ  = collect(range(T(-9) / 10, T(19) / 20; length = n - 1))
    Py = q * adjoint(q) + Q0 * Diagonal(λ) * adjoint(Q0)
    G  = L * Py * adjoint(L)

    out = reduced_resolvent_norm(s.M, G, s.m)

    # I - P is normal with spectrum {0} ∪ {1 - λ_k}, so the singular values on
    # the zero-mass space are the |1 - λ_k| and the answer is exactly the
    # reciprocal of the spectral gap.
    @test out.norm ≈ inv(minimum(abs.(one(T) .- λ))) rtol=rn_rtol(T)
    @test out.mass_error ≤ mass_tolerance(T)
end


@testset "RN-6 invariance under severe basis rescaling ($T)" for T in (Float64, Double64)
    s = RN[T]
    n = length(s.m)

    # Rescale the basis, φ_i → 10^{k_i} φ_i with k spread over [-6, 6]:
    #
    #     M̃ = S M S,    G̃ = S G S,    m̃ = S m,      S = diag(10^{k_i}).
    #
    # V_N is unchanged, so M_{ε,N} must be unchanged too — this is the coordinate
    # invariance that justifies equilibrating at all.  But κ(M̃) ~ 10^24 κ(M),
    # far past 1/eps(Float64), so the unscaled route of RN-1 is now hopeless while
    # the equilibrated one is untouched: D̃ = D S⁻¹ exactly, hence M̃_s = D̃ M̃ D̃ =
    # M_s.  Powers of ten (rather than two) are used deliberately — they do not
    # rescale exactly in binary, so forming M̃ perturbs every entry and the
    # recovery has to be genuinely stable rather than bit-trivial.
    S  = Diagonal(T(10) .^ collect(range(T(-6), T(6); length = n)))
    Mt = Symmetric(S * s.M * S)     # wrapped: S*M*S is symmetric only to roundoff
    Gt = S * s.G * S
    mt = S * s.m

    out = reduced_resolvent_norm(Mt, Gt, mt)

    @test out.norm      ≈ s.out.norm      rtol=rn_rtol(T)
    @test out.sigma_min ≈ s.out.sigma_min rtol=rn_rtol(T)

    # The raw Gram matrix is now catastrophically conditioned; the equilibrated
    # one is exactly as well conditioned as before.  This is the whole point.
    @test out.cond_mass > 1e20
    @test out.cond_mass_scaled ≈ s.out.cond_mass_scaled rtol=1e-8
    @test out.mass_error ≤ mass_tolerance(T)
end


@testset "RN-7 BigFloat generic linear algebra" begin
    # The implementation claims full precision for any T with generic cholesky,
    # qr and svdvals.  Float64 and Double64 are covered above; this confirms the
    # BigFloat path actually runs, on a small synthetic problem with an exact
    # known answer.  Any SPD matrix is the Gram matrix of some basis, so a Hilbert
    # block (κ ≈ 5e5, not trivially conditioned) is a legitimate stand-in.
    setprecision(BigFloat, 256) do
        T = BigFloat
        n = 5

        M = T[1 // (i + j - 1) for i in 1:n, j in 1:n]
        m = T[1 // i for i in 1:n]

        # Same self-adjoint construction as RN-5; dyadic λ so they are exact.
        L  = cholesky(Symmetric(M)).L
        q  = normalize(L \ m)
        Q0 = orthonormal_complement(q)
        λ  = T[-3//4, -1//4, 1//4, 5//8]
        G  = L * (q * adjoint(q) + Q0 * Diagonal(λ) * adjoint(Q0)) * adjoint(L)

        out = reduced_resolvent_norm(M, G, m)

        @test out.norm ≈ inv(minimum(abs.(one(T) .- λ))) rtol=T(1e-60)
        @test out.mass_error ≤ T(1e-60)
        @test out.norm isa BigFloat          # nothing was narrowed to Float64
    end
end
