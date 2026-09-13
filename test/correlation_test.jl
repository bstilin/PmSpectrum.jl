# -------------------------------------------------
# Tests for deterministic autocorrelation curves
# -------------------------------------------------
#
# correlation_curve evaluates
#
#     C_ε(n) = ∫ f₀(x) · P_ε^n(f₀ ρ_ε)(x) dx,     f₀ = f - ∫ f ρ_ε
#
# from the Galerkin matrices.  What is tested:
#
#   CR-1  transfer-operator orientation: one application is M \ (G a), not M \ (G' a)
#   CR-2  normalisation of ρ_N and centering of f₀
#   CR-3  the moment functionals ℓ_j = ∫ f₀ φ_j and b_i = ∫ φ_i f₀ ρ_N vs quadgk
#   CR-4  zero mass of f₀ρ_N, preserved exactly under iteration
#   CR-5  C_N(0) against the directly integrated invariant variance
#   CR-6  direct iteration against the eigen-decomposition of the same operator
#   CR-7  convergence with respect to basis resolution
#   CR-8  long-lag decay against the independently computed eigenvalues
#
# ε is deliberately large enough (0.05) that the discretised operator has a
# genuinely isolated spectral gap; at the ε ~ 1e-6 used elsewhere in the suite
# the subleading eigenvalues form a cluster near 1 and CR-8 would be meaningless.

using Test
using PmSpectrum
using PmSpectrum.Bases: build_single_splines, density_quadrature,
    integrate_against_density, EndpointRefinement, correlation_curve,
    estimate_decay_rate, _correlation_setup
using PmSpectrum.Utils: scaled_nonsymmetric_eigen
using LinearAlgebra: cholesky, Symmetric, norm
using QuadGK: quadgk


# One shared solve, reused by every set below.  N = 80 keeps it cheap; CR-7 adds
# the only second solve.
const CR_ALPHA  = 0.5
const CR_EPS    = 0.05
const CR_MAXLAG = 40

const cr_result = run_single_experiment(CR_ALPHA, CR_EPS, 80, 32)
const cr_rh     = rehydrate(cr_result)
const cr_spls   = build_single_splines(cr_rh.basis)
const cr_mass   = [ϕ.mass for ϕ in cr_spls]
const cr_lambda = complex.(cr_result.eigenvalues_re, cr_result.eigenvalues_im)

# The map is symmetric about 1/2, so eigenfunctions split into even and odd
# classes about 1/2.  cos(2πx) is even there and sin(2πx) is odd, so the mixed
# observable excites the leading mode of each class.
cr_mixed(x) = cos(2π * x) + 0.5 * sin(2π * x)
cr_even(x)  = cos(2π * x)
cr_odd(x)   = sin(2π * x)

const cr_curve = correlation_curve(cr_mixed, cr_rh, CR_MAXLAG)


"""
    cr_spectral_decomposition(f, rh, max_lag; n_quad = 32)

Reference implementation of the same curve through the eigen-decomposition of the
discrete operator, used by CR-6 and CR-8 as an independent second evaluation.

Writing `a₀ = Σ_j y_j v_j` in the basis of right eigenvectors of `G v = λ M v`,

    C_N(n) = Σ_j A_j λ_jⁿ,     A_j = (ℓᵀ v_j) · y_j,
"""
function cr_spectral_decomposition(
    f,
    rh      :: RehydratedResult{T},
    max_lag :: Int;
    n_quad  :: Int = 32,
) where {T<:AbstractFloat}

    r    = rh.result
    spls = build_single_splines(rh.basis)

    ℓ, b, _, _, _, _ = _correlation_setup(
        f, spls, r.c, r.epsilon;
        n_quad = n_quad, refine = EndpointRefinement(r.alpha))

    a0 = cholesky(Symmetric(r.M)) \ b

    λ_raw, V_raw = scaled_nonsymmetric_eigen(r.G, r.M)
    perm = sortperm(abs.(λ_raw); rev = true)
    λ    = λ_raw[perm]
    V    = V_raw[:, perm]

    y = V \ Complex{T}.(a0)
    A = (transpose(V) * Complex{T}.(ℓ)) .* y

    Cc = Vector{Complex{T}}(undef, max_lag + 1)
    z  = copy(A)
    for k in 0:max_lag
        Cc[k+1] = sum(z)
        k == max_lag && break
        z .*= λ
    end

    return (C             = real.(Cc),
            eigenvalues   = λ,
            amplitudes    = A,
            imag_residual = maximum(abs, imag.(Cc)))
end


@testset "CR-1 transfer-operator orientation" begin
    F = cholesky(Symmetric(cr_result.M))
    rel(v) = norm(v - cr_result.c) / norm(cr_result.c)

    # ρ_N is the fixed point of the discrete operator, so the correct coefficient
    # action reproduces it.
    @test rel(F \ (cr_result.G * cr_result.c)) < 1e-12

    # ... and the transpose emphatically does not.  This is the guard that fails
    # loudly if the G[j,i] = ⟨P_ε φ_i, φ_j⟩ convention is ever misread.
    @test rel(F \ (transpose(cr_result.G) * cr_result.c)) > 1e-3

    # Partition of unity: 1 ∈ span{φ_j}, so Σ_j G[j,i] = ∫ P_ε φ_i = ∫ φ_i.  This
    # is what makes the discrete operator mass-preserving (CR-4).
    @test vec(sum(cr_result.G, dims = 1)) ≈ cr_mass rtol = 1e-12
end


@testset "CR-2 normalisation and centering" begin
    @test cr_curve.density_mass ≈ 1.0 atol = 1e-13

    # ∫ f₀ ρ_N.  Because 1 ∈ span{φ_j}, the mass of the *projected* h₀ equals the
    # mass of h₀ itself, so mass[1] is exactly the centering residual.
    @test abs(cr_curve.mass[1]) < 1e-14

    # m_N agrees with the existing observable-integration path.
    dq = density_quadrature(cr_rh; n_quad = 32)
    @test cr_curve.mean ≈ integrate_against_density(cr_mixed, dq) rtol = 1e-13

    # And the invariant density is the one the run normalised.
    @test sum(cr_result.c .* cr_mass) ≈ 1.0 rtol = 1e-14
end


@testset "CR-3 moment functionals" begin
    ℓ, b, mvec, mean, variance, density_mass = _correlation_setup(
        cr_mixed, cr_spls, cr_result.c, CR_EPS;
        n_quad = 32, refine = EndpointRefinement(CR_ALPHA))

    @test mvec ≈ cr_mass
    @test mean == cr_curve.mean
    @test density_mass == cr_curve.density_mass

    f0(x) = cr_mixed(x) - mean

    # quadgk with the spline's own knots as breakpoints, the reference style used
    # throughout this suite.  A few j spread across the basis, including the ones
    # adjacent to the endpoint singularities.
    for j in (1, 2, 5, 20, length(cr_spls) ÷ 2, length(cr_spls) - 1, length(cr_spls))
        ϕ      = cr_spls[j]
        ref_ℓ  = 0.0
        ref_b  = 0.0
        for (sa, sb) in ϕ.support
            brk = unique!(sort!(vcat(sa, filter(k -> sa < k < sb, ϕ.knots), sb)))
            ref_ℓ += quadgk(x -> f0(x) * ϕ(x),                brk...; rtol = 1e-13)[1]
            ref_b += quadgk(x -> f0(x) * ϕ(x) * cr_rh.pdf(x), brk...; rtol = 1e-13)[1]
        end
        @test ℓ[j] ≈ ref_ℓ rtol = 1e-10
        @test b[j] ≈ ref_b rtol = 1e-10
    end

    # ∫ f₀² ρ_N by an independent adaptive rule, with the breakpoints supplied so
    # quadgk does not have to discover the boundary layer on its own.
    ref_var = quadgk(x -> f0(x)^2 * cr_rh.pdf(x), cr_result.break_points...; rtol = 1e-12)[1]
    @test variance ≈ ref_var rtol = 1e-10
end


@testset "CR-4 zero mass, preserved under iteration" begin
    # ∫ h₀ = ∫ f₀ ρ_N = 0 by construction, and the discrete operator preserves it
    # exactly (CR-1), so this is zero at every lag to the accuracy of the assembled
    # G — which in Float64 is round-off.  (In Double64 the floor is the operator's
    # own accuracy, ~1e-17, not eps(T); see the CorrelationCurve docstring.)
    @test maximum(abs, cr_curve.mass) < 1e-14
    @test length(cr_curve.mass) == CR_MAXLAG + 1
    @test cr_curve.lags == 0:CR_MAXLAG
end


@testset "CR-5 C_N(0) vs the invariant variance" begin
    # C_N(0) = ⟨f₀, Π_N h₀⟩ differs from ∫f₀²ρ_N by ⟨f₀ - Π_N f₀, h₀ - Π_N h₀⟩,
    # the product of two projection errors, so for a smooth f the two agree far
    # more tightly than either does with the continuum limit.
    @test cr_curve.C[1] ≈ cr_curve.variance rtol = 1e-10
    @test cr_curve.C[1] > 0
end


@testset "CR-6 iteration vs spectral decomposition" begin
    sd = cr_spectral_decomposition(cr_mixed, cr_rh, CR_MAXLAG)

    # Same eigenvalues, same ordering, as the stored run.
    @test sd.eigenvalues ≈ cr_lambda

    # The two independent evaluations of the same curve.
    @test maximum(abs, sd.C .- cr_curve.C) < 1e-10 * abs(cr_curve.C[1])
    @test sd.imag_residual < 1e-12

    # h₀ has no component on the invariant mode: the mass functional is the left
    # eigenvector at λ = 1, and ∫h₀ = 0.
    @test abs(sd.amplitudes[1]) < 1e-14

    # Σ_j A_j = C_N(0).
    @test real(sum(sd.amplitudes)) ≈ cr_curve.C[1] rtol = 1e-10
end


@testset "CR-7 convergence in basis resolution" begin
    result_fine = run_single_experiment(CR_ALPHA, CR_EPS, 140, 32)
    curve_fine  = correlation_curve(cr_mixed, result_fine, CR_MAXLAG)

    @test maximum(abs, curve_fine.C .- cr_curve.C) < 1e-6 * abs(cr_curve.C[1])
    @test curve_fine.mean ≈ cr_curve.mean rtol = 1e-6
    @test curve_fine.variance ≈ cr_curve.variance rtol = 1e-6
    @test curve_fine.density_mass ≈ 1.0 atol = 1e-13
    @test maximum(abs, curve_fine.mass) < 1e-14
end


@testset "CR-8 long-lag decay vs the computed spectrum" begin
    λ2, λ3 = abs(cr_lambda[2]), abs(cr_lambda[3])
    @test λ2 < 1 - 1e-3          # the gap is genuinely isolated at this ε
    @test λ3 < λ2

    # cos(2πx) is even about 1/2 and therefore couples to the even class only:
    # a single geometric mode, so the fitted rate is |λ₂| to round-off.
    curve_even = correlation_curve(cr_even, cr_rh, CR_MAXLAG)
    sd_even    = cr_spectral_decomposition(cr_even, cr_rh, CR_MAXLAG)
    @test estimate_decay_rate(curve_even).rate ≈ λ2 rtol = 1e-8
    @test abs(sd_even.amplitudes[2]) > 0.1
    @test abs(sd_even.amplitudes[3]) < 1e-14        # odd mode not excited

    # sin(2πx) is odd about 1/2, so it skips λ₂ entirely and decays at |λ₃|.
    curve_odd = correlation_curve(cr_odd, cr_rh, CR_MAXLAG)
    sd_odd    = cr_spectral_decomposition(cr_odd, cr_rh, CR_MAXLAG)
    @test estimate_decay_rate(curve_odd).rate ≈ λ3 rtol = 1e-8
    @test abs(sd_odd.amplitudes[3]) > 0.1
    @test abs(sd_odd.amplitudes[2]) < 1e-14         # even mode not excited

    # The mixed observable carries both; the slower one wins the tail, with the
    # faster one still contributing a small correction at lag 40.
    @test estimate_decay_rate(cr_curve).rate ≈ λ2 rtol = 1e-3
    @test cr_curve.C[1] ≈ curve_even.C[1] + 0.25 * curve_odd.C[1] rtol = 1e-10
end
