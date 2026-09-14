# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

# -------------------------------------------------
# Tests for integration of observables against the invariant density
# -------------------------------------------------
#
# The quadrature mesh (build_integration_mesh) is build_star_partition on the
# identity branch, so its S* properties are already covered by SP-1..SP-5 and
# SS-0..SS-4 in bspline_galerkin_matrix_test.jl. What is tested here is the
# integration layer built on top of it:
#
#   DI-1  mesh structure: endpoints, sortedness, knot inclusion, ±ε shifts,
#         dyadic endpoint refinement
#   DI-2  unit mass: ∫ρ = 1, cross-checked against Σ c_i ∫φ_i
#   DI-3  fractional endpoint behaviour: ∫₀¹ x^β dx = 1/(1+β) for β > 0 to
#         round-off, and a *negative* β improving monotonically with τ_end
#   DI-4  independent reference: agreement with adaptive quadgk
#   DI-5  n_quad self-convergence
#   DI-6  doubling-map anchor: λ(α) → log 2 as α → 0
#   DI-7  symmetric_pm_derivative / log_symmetric_pm_derivative

using Test
using PmSpectrum
using PmSpectrum.Bases: build_integration_mesh, integrate_on_mesh,
    density_quadrature, integrate_against_density, total_mass,
    EndpointRefinement, build_single_splines
using PmSpectrum.Utils: symmetric_pm, symmetric_pm_derivative,
    log_symmetric_pm_derivative
using BSplineKit: knots as bsk_knots
using QuadGK: quadgk


# One shared, moderately sized solve reused by most of the sets below.
const DI_ALPHA = 0.5
const DI_EPS   = 1e-6
const di_result = run_single_experiment(DI_ALPHA, DI_EPS, 100, 32)
const di_rh     = rehydrate(di_result)
const di_knots  = collect(Float64, bsk_knots(di_rh.basis))


@testset "DI-1 mesh structure" begin
    ε    = DI_EPS
    atol = eps(Float64)
    mesh = build_integration_mesh(di_knots, ε; refine = EndpointRefinement(DI_ALPHA))

    @test mesh[1]   == 0.0
    @test mesh[end] == 1.0
    @test issorted(mesh)
    @test all(diff(mesh) .> atol)          # dedup left no degenerate panels

    # Every base knot inside [0,1] survives into the mesh.
    for p in di_knots
        0 <= p <= 1 || continue
        @test minimum(abs.(mesh .- p)) <= atol
    end

    # The ±ε shifted breakpoints are present.
    for p in di_knots
        0 <= p <= 1 || continue
        for s in (mod(p + ε, 1.0), mod(p - ε, 1.0))
            @test minimum(abs.(mesh .- s)) <= atol
        end
    end

    # Dyadic refinement fired at BOTH endpoints. The identity branch owns both
    # 0 and 1, unlike either PM branch, so this is the case that catches the
    # left grid being appended before the right reflection reads P.
    plain_mesh = build_integration_mesh(di_knots, ε)
    @test mesh[2] < plain_mesh[2]                        # deeper on the left
    @test 1 - mesh[end-1] < 1 - plain_mesh[end-1]        # and on the right

    # Base knots and ±ε shifts are symmetric under x → 1-x, so a correct
    # refinement leaves the mesh reaching comparably close to either endpoint.
    @test 1 - mesh[end-1] < 10 * mesh[2]
    @test mesh[2] < 10 * (1 - mesh[end-1])

    # Without refinement there is nothing below the smallest shifted knot.
    plain = build_integration_mesh(di_knots, ε)
    @test length(plain) < length(mesh)
    @test minimum(filter(>(0), mesh)) < minimum(filter(>(0), plain))
end


@testset "DI-2 unit mass" begin
    dq = density_quadrature(di_rh; n_quad = 64)

    @test total_mass(dq) ≈ 1.0 atol = 1e-13
    @test integrate_against_density(x -> 1.0, dq) ≈ 1.0 atol = 1e-13

    # Cross-check against the coefficient-based mass used to normalise the
    # eigenvector in run_single_experiment.
    spls = build_single_splines(di_rh.basis)
    m_c  = sum(di_result.c[i] * spls[i].mass for i in eachindex(di_result.c))
    @test total_mass(dq) ≈ m_c atol = 1e-13

    # A DensityQuadrature built on the constant density 1 is the bare rule.
    dq1 = density_quadrature(x -> 1.0, di_knots, DI_EPS;
                             n_quad = 32, refine = EndpointRefinement(DI_ALPHA))
    @test total_mass(dq1) ≈ 1.0 atol = 1e-14
end


@testset "DI-3 fractional endpoint behaviour" begin
    mesh = build_integration_mesh(di_knots, DI_EPS;
                                  refine = EndpointRefinement(DI_ALPHA))

    # Positive fractional powers (the log T' case: bounded, cusped derivative)
    # are resolved to round-off by the geometric endpoint grid.
    for β in (0.15, 0.5, 0.85, 1.5)
        @test integrate_on_mesh(x -> x^β, mesh; n_quad = 64) ≈ 1 / (1 + β) atol = 1e-13
        @test integrate_on_mesh(x -> (1 - x)^β, mesh; n_quad = 64) ≈ 1 / (1 + β) atol = 1e-13
    end

    # A genuinely singular (but integrable) observable is limited by how close
    # to the endpoint the mesh reaches, so deepening τ_end must improve it.
    # The Float64 floor is the atol dedup at x ≈ eps, hence the modest target.
    exact = 2.0                                  # ∫₀¹ x^(-1/2) dx
    f     = x -> x <= 0 ? 0.0 : x^(-0.5)
    errs  = map((1e-12, 1e-20, 1e-30)) do τ
        m = build_integration_mesh(di_knots, DI_EPS;
                                   refine = EndpointRefinement(DI_ALPHA; τ_end = τ))
        abs(integrate_on_mesh(f, m; n_quad = 64) - exact)
    end
    @test errs[end] < errs[1] / 100              # deeper grid buys real accuracy
    @test errs[end] < 1e-8
end


@testset "DI-4 independent reference (quadgk)" begin
    dq = density_quadrature(di_rh; n_quad = 64)

    # Give quadgk the same panel boundaries so it does not have to discover the
    # boundary layer by itself; the *rule* is then genuinely independent.
    segs = dq.mesh
    for f in (x -> log_symmetric_pm_derivative(x, DI_ALPHA),
              x -> x^2,
              x -> sinpi(3x))
        ref, _ = quadgk(x -> f(x) * di_rh.pdf(x), segs...; rtol = 1e-13)
        @test integrate_against_density(f, dq) ≈ ref rtol = 1e-11
    end
end


@testset "DI-5 n_quad self-convergence" begin
    f  = x -> log_symmetric_pm_derivative(x, DI_ALPHA)
    λs = [integrate_against_density(f, density_quadrature(di_rh; n_quad = nq))
          for nq in (16, 32, 64, 128)]

    @test all(isfinite, λs)
    @test λs[2] ≈ λs[4] rtol = 1e-11
    @test λs[3] ≈ λs[4] rtol = 1e-12
end


@testset "DI-6 doubling-map anchor" begin
    # As α → 0 the map degenerates to T(x) = 2x on [0,1/2] (the doubling map),
    # whose invariant density is 1 and whose Lyapunov exponent is log 2. This
    # pins the observable AND the density convention: the eigenvector solves
    # G_ε P ρ = ρ, so ρ is the stationary density at the point where T' is
    # evaluated, and no correction factor belongs in the Birkhoff average.
    α  = 0.05
    r  = run_single_experiment(α, 1e-8, 120, 32)
    dq = density_quadrature(rehydrate(r); n_quad = 64)
    λ  = integrate_against_density(x -> log_symmetric_pm_derivative(x, α), dq)

    @test λ ≈ log(2) rtol = 5e-3

    # λ decreases with α: more time spent near the neutral fixed points.
    λs = map((0.15, 0.5, 0.85)) do a
        rr = run_single_experiment(a, 1e-8, 120, 32)
        integrate_against_density(x -> log_symmetric_pm_derivative(x, a),
                                  density_quadrature(rehydrate(rr); n_quad = 64))
    end
    @test all(λs .> 0)
    @test issorted(λs; rev = true)
    @test λs[1] < λ
end


@testset "DI-7 map derivative" begin
    α = 0.37
    h = 1e-6

    # Central difference away from the endpoints and the midpoint kink.
    for x in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)
        fd = (symmetric_pm(x + h, α) - symmetric_pm(x - h, α)) / (2h)
        @test symmetric_pm_derivative(x, α) ≈ fd rtol = 1e-7
    end

    # Symmetry T'(x) = T'(1-x), and T' ≥ 1 everywhere (so log needs no abs).
    for x in (0.0, 1e-12, 0.1, 0.5, 0.9, 1.0)
        @test symmetric_pm_derivative(x, α) ≈ symmetric_pm_derivative(1 - x, α)
        @test symmetric_pm_derivative(x, α) >= 1
        @test log_symmetric_pm_derivative(x, α) ≈ log(symmetric_pm_derivative(x, α)) rtol = 1e-13
    end

    # ... and where the two forms disagree, log1p is the accurate one. Near the
    # endpoints the argument of log1p is small, which is exactly the regime the
    # boundary-layer runs care about.
    let x = 1e-12
        truth = Float64(log1p((big(1) + big(α)) * exp2(big(α)) * big(x)^big(α)))
        e_log1p = abs(log_symmetric_pm_derivative(x, α) - truth)
        e_naive = abs(log(symmetric_pm_derivative(x, α)) - truth)
        @test e_log1p < e_naive
    end

    # Endpoint limit: T'(0) = 1, so log T'(0) = 0.
    @test symmetric_pm_derivative(0.0, α) == 1.0
    @test log_symmetric_pm_derivative(0.0, α) == 0.0

    @test_throws ArgumentError symmetric_pm_derivative(-0.1, α)
    @test_throws ArgumentError log_symmetric_pm_derivative(1.1, α)

    # Double64 stays in Double64.
    @test symmetric_pm_derivative(Double64(0.3), Double64(α)) isa Double64
    @test log_symmetric_pm_derivative(Double64(0.3), Double64(α)) isa Double64
end
