# -------------------------------------
# Tests for TransferPrediction.jl
#
# Sturm-Liouville eigenpairs (λ_j, v_j)  ->  predicted transfer-operator modes
#   Λ_{ε,j} ≈ exp(-λ_j L_ε^α),   φ_{ε,j}(x) = L_ε⁻¹ e^{Ψ(x/L_ε)} v_j(x/L_ε)
#
#   TP-1   psi_weight formula, parity, genericity
#   TP-2   the eigenvalue map and its two forms
#   TP-2b  the two ε conventions agree (the √3)
#   TP-3   asymptotic gap 1-Λ ~ λ L_ε^α
#   TP-3b  predicted / recovered round trip
#   TP-4   extend_parity
#   TP-5   the three eigenfunction variables
#   TP-6   eigenfunction prediction vs a measured transfer eigenfunction
#   TP-7   eigenvalue prediction vs a measured transfer spectrum
#   TP-8   psi_cutoff
# -------------------------------------


using DoubleFloats
using BSplineKit
using PmSpectrum.TransferPrediction: psi_weight, psi_cutoff, extend_parity,
    layer_scale, sd_from_halfwidth, halfwidth_from_sd,
    predicted_eigenvalue, predicted_eigenvalues,
    recovered_eigenvalue, recovered_eigenvalues,
    predicted_eigenfunction, predict_transfer_mode, predict_transfer_spectrum
using PmSpectrum.SchrodingerFD: solve_fd
using PmSpectrum.SchrodingerShooting: refine_eigenvalue, shooting_eigenfunction,
    turning_point, bracket_from_fd
using PmSpectrum.Utils: boundary_layer_scale, scaled_nonsymmetric_eigen


######################################################
# The weight
######################################################

@testset "psi_weight: formula, parity, genericity (TP-1)" begin
    for α in (0.15, 0.5, 0.85), y in (0.0, 0.3, 1.0, 4.7)
        @test psi_weight(y, α) ≈ 2.0^(α - 1) * abs(y)^(2 + α) / (2 + α) rtol=1e-15 atol=1e-300
        @test psi_weight(-y, α) == psi_weight(y, α)          # even in y
    end
    @test psi_weight(0.0, 0.5) == 0.0

    @test psi_weight(Double64(0.3), Double64(0.5)) isa Double64
    @test psi_weight(big(0.3), big(0.5)) isa BigFloat
    @test Float64(psi_weight(Double64(0.3), Double64(0.5))) ≈ psi_weight(0.3, 0.5) rtol=1e-15
end


######################################################
# Eigenvalues
######################################################

@testset "eigenvalue map: forms, range, ordering (TP-2)" begin
    α, Lε = 0.5, 0.01
    λs = [1.51, 3.70, 5.96, 8.47]

    Λs = predicted_eigenvalues(λs, α, Lε)
    @test all(0 .< Λs .< 1)
    @test issorted(Λs; rev=true)          # larger λ ⇒ smaller multiplier

    # :linear is the linearisation, so the two agree to O((λL^α)²)
    for λ in λs
        s = λ * Lε^α
        e = predicted_eigenvalue(λ, α, Lε; form=:exponential)
        l = predicted_eigenvalue(λ, α, Lε; form=:linear)
        @test l ≈ 1 - s rtol=1e-14
        @test abs(e - l) < s^2            # difference is second order
    end

    # ...and that difference is asymptotically exactly s²/2, since
    # exp(-s) - (1-s) = s²/2 - s³/6 + …
    d(Lε) = abs(predicted_eigenvalue(λs[1], α, Lε; form=:exponential) -
                predicted_eigenvalue(λs[1], α, Lε; form=:linear))
    for Lε in (1e-4, 1e-6, 1e-8)
        s = λs[1] * Lε^α
        @test d(Lε) ≈ s^2 / 2 rtol=s        # the s³/6 term is the O(s) correction
    end
    @test d(1e-8) < d(1e-6) / 50            # d ∝ L_ε^{2α}: a factor 100 per 100 in L_ε

    @test_throws ArgumentError predicted_eigenvalue(1.5, α, Lε; form=:quadratic)
    @test_throws ArgumentError recovered_eigenvalue(0.9, α, Lε; form=:quadratic)
end


@testset "the two ε conventions agree (TP-2b)" begin
    # layer_scale takes the STANDARD DEVIATION; Utils.boundary_layer_scale takes
    # the HALF-WIDTH. They are the same number because ε = ε_hw/√3 gives
    # ε²/2 = ε_hw²/6 identically -- this is an identity, so it must hold to
    # rounding, and it is the one check that catches a lost or inverted √3.
    for α in (0.15, 0.5, 0.85), ε_hw in (1e-2, 1e-6, 1e-12, 1e-17)
        @test layer_scale(α, sd_from_halfwidth(ε_hw)) ≈ boundary_layer_scale(α, ε_hw) rtol=1e-15
    end

    for ε in (1e-2, 1e-7, 1e-15)
        @test halfwidth_from_sd(sd_from_halfwidth(ε)) ≈ ε rtol=1e-15
        @test sd_from_halfwidth(halfwidth_from_sd(ε)) ≈ ε rtol=1e-15
    end
    @test halfwidth_from_sd(1.0) ≈ sqrt(3) rtol=1e-15

    # The failure mode, recorded executably: feeding a half-width straight to
    # layer_scale is wrong by exactly 3^(1/(2+α)) -- ~1.55 at α=0.5, silent.
    for α in (0.15, 0.5, 0.85)
        ε_hw = 1e-6
        @test layer_scale(α, ε_hw) / layer_scale(α, sd_from_halfwidth(ε_hw)) ≈
              3^(1 / (2 + α)) rtol=1e-14
    end

    @test layer_scale(Double64(1)/2, Double64(1e-6)) isa Double64
end


@testset "asymptotic gap 1-Λ ~ λ L_ε^α (TP-3)" begin
    α, λ = 0.5, 1.51
    Lεs = (1e-2, 1e-4, 1e-6, 1e-8)
    ratios = map(Lεs) do Lε
        m = predict_transfer_mode(λ, identity, α, Lε)
        m.gap / m.linearized_gap
    end
    @test all(ratios .< 1)                # 1-exp(-s) < s, so gap < linearized gap
    @test issorted(ratios)                # ...approaching 1 from below

    # the approach is at a definite rate: (1-exp(-s))/s = 1 - s/2 + O(s²)
    for (Lε, r) in zip(Lεs, ratios)
        s = λ * Lε^α
        @test 1 - r ≈ s / 2 rtol=s
    end
    @test ratios[end] ≈ 1 rtol=1e-3       # s/2 ≈ 7.6e-5 at the smallest L_ε
end


@testset "predicted / recovered round trip (TP-3b)" begin
    α = 0.5
    for form in (:exponential, :linear), Lε in (1e-2, 1e-6), λ in (1.51, 8.47, 25.4)
        Λ = predicted_eigenvalue(λ, α, Lε; form=form)
        @test recovered_eigenvalue(Λ, α, Lε; form=form) ≈ λ rtol=1e-12
    end

    λs = [1.51, 3.70, 5.96]
    @test recovered_eigenvalues(predicted_eigenvalues(λs, α, 1e-4), α, 1e-4) ≈ λs rtol=1e-12

    # the invariant density carries no Sturm-Liouville eigenvalue
    @test recovered_eigenvalue(1.0, α, 1e-4) == 0.0
end


######################################################
# Eigenfunctions
######################################################

@testset "extend_parity (TP-4)" begin
    f(y) = exp(-y^2) * (1 + y)                       # arbitrary half-line function
    fe, fo = extend_parity(f, :even), extend_parity(f, :odd)
    for y in (0.3, 1.0, 2.5)
        @test fe(-y) == fe(y)
        @test fo(-y) == -fo(y)
        @test fe(y) == f(y) && fo(y) == f(y)
    end
    @test fo(0.0) == 0.0                             # sign(0) == 0: the odd BC
    @test_throws ArgumentError extend_parity(f, :sideways)
end


@testset "the three eigenfunction variables (TP-5)" begin
    α, Lε = 0.5, 0.05
    v(y) = exp(-psi_weight(y, α))          # decays like a real eigenfunction

    Φ = predicted_eigenfunction(v, α, Lε; variable=:rescaled)
    φ̃ = predicted_eigenfunction(v, α, Lε; variable=:inner)
    φ = predicted_eigenfunction(v, α, Lε; variable=:outer)

    for y in (-2.0, -0.4, 0.0, 0.4, 2.0)
        @test φ̃(y) ≈ Φ(y) / Lε rtol=1e-14           # inner is rescaled / L_ε
        @test φ(Lε * y) ≈ φ̃(y) rtol=1e-14           # outer at x = L_ε y is inner at y
    end

    # with v = e^{-Ψ} exactly, the weight cancels and Φ ≡ 1
    for y in (-1.5, 0.0, 1.5)
        @test Φ(y) ≈ 1.0 rtol=1e-13
    end

    @test_throws ArgumentError predicted_eigenfunction(v, α, Lε; variable=:sideways)
end


######################################################
# Against measured transfer-operator data
######################################################

@testset "eigenfunction prediction vs measurement (TP-6)" begin
    # Forward direction of the earlier experiment: build Φ_0 = e^Ψ v_0 from the
    # Sturm-Liouville ground state and compare with the SECOND eigenfunction of a
    # real transfer-operator solve, in inner variables.
    α, L, ε_hw = 0.5, 8.0, 1e-8
    Lε = layer_scale(α, sd_from_halfwidth(ε_hw))

    fd = solve_fd(α, L, 4000; parity=:even, nev=2)
    sh = refine_eigenvalue(α, bracket_from_fd(fd.eigenvalues, 1), L,
                           turning_point(α, fd.eigenvalues[1]); parity=:even)
    ef = shooting_eigenfunction(α, sh.eigenvalue, L, turning_point(α, fd.eigenvalues[1]);
                                parity=:even, dtmax=0.01)
    v0 = extend_parity(ef.eigenfunction, :even)
    Φ0 = predicted_eigenfunction(v0, α, Lε; variable=:rescaled)

    res  = run_single_experiment(α, ε_hw, 100, 32)
    λt, V = scaled_nonsymmetric_eigen(res.G, res.M)
    perm  = sortperm(real.(λt); rev=true)
    B     = BSplineKit.BSplineBasis(BSplineOrder(res.degree + 1), copy(res.break_points))
    φ2    = BSplineKit.Spline(B, real.(V[:, perm[2]]))

    # stay well inside where e^Ψ v is trustworthy
    ys = range(0.05, 4.0, length=120)
    @test 4.0 < psi_cutoff(α, 1e-12)

    pred = [Φ0(y)         for y in ys]
    meas = [φ2(y * Lε)    for y in ys]
    pred ./= maximum(abs, pred)
    meas ./= maximum(abs, meas)
    meas .*= sign(sum(pred .* meas))          # eigenvector sign is arbitrary

    @test maximum(abs, pred .- meas) < 5e-3
end


@testset "eigenvalue prediction vs measurement (TP-7)" begin
    # Λ_{ε,j} is the j-th largest NONSTATIONARY eigenvalue, so after sorting by
    # decreasing real part it is entry j+2 -- entry 1 is the invariant density,
    # with Λ = 1. That offset is the most error-prone part of using this module.
    α, L, ε_hw = 0.5, 8.0, 1e-8
    Lε = layer_scale(α, sd_from_halfwidth(ε_hw))

    sectors = map((:even, :odd)) do parity
        fd = solve_fd(α, L, 4000; parity=parity, nev=2)
        [refine_eigenvalue(α, bracket_from_fd(fd.eigenvalues, j), L,
                           turning_point(α, fd.eigenvalues[j]); parity=parity).eigenvalue
         for j in 1:2]
    end
    λ = sort(vcat(sectors...))                 # full-line merged ordering

    res  = run_single_experiment(α, ε_hw, 100, 32)
    meas = sort(res.eigenvalues_re; rev=true)
    @test meas[1] ≈ 1.0 atol=1e-8              # the invariant density

    for j in 0:2
        Λp = predicted_eigenvalue(λ[j+1], α, Lε)
        Λm = meas[j+2]                         # the offset
        @test Λp ≈ Λm rtol=1e-2                # asymptotic relation, loose on purpose
        # recovering λ from the measurement lands near the Sturm-Liouville value
        @test recovered_eigenvalue(Λm, α, Lε) ≈ λ[j+1] rtol=1e-2
    end
end


@testset "psi_cutoff (TP-8)" begin
    for α in (0.15, 0.5, 0.85), tol in (1e-8, 1e-15, 1e-25)
        y = psi_cutoff(α, tol)
        @test psi_weight(y, α) ≈ -log(tol) rtol=1e-12
    end
    # tighter tolerance ⇒ larger usable radius
    @test psi_cutoff(0.5, 1e-25) > psi_cutoff(0.5, 1e-15) > psi_cutoff(0.5, 1e-8)
    @test psi_cutoff(0.5, 1e-15) ≈ 6.8 rtol=5e-2       # the α=0.5 figure quoted in the docs

    @test_throws ArgumentError psi_cutoff(0.5, 0.0)
    @test_throws ArgumentError psi_cutoff(0.5, 2.0)

    # beyond the cutoff, e^Ψ amplifies v's noise: perturbing v at the 1e-15 level
    # is O(1) in Φ there, and negligible well inside.
    α = 0.5
    v_exact(y)   = exp(-psi_weight(y, α))
    v_noisy(y)   = v_exact(y) + 1e-15
    Φe, Φn = (predicted_eigenfunction(f, α, 1.0; variable=:rescaled) for f in (v_exact, v_noisy))
    yc = psi_cutoff(α, 1e-15)
    @test abs(Φn(0.5 * yc) - Φe(0.5 * yc)) < 1e-3
    @test abs(Φn(1.2 * yc) - Φe(1.2 * yc)) > 1.0
end
