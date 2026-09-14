# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

# -------------------------------------
# Tests for Results.jl
# -------------------------------------


using PmSpectrum
using BSplineKit
using DoubleFloats
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Build a small but realistic InvariantDensityResult at type T
function make_result(::Type{T}; decouple::Bool=false) where {T<:AbstractFloat}
    alpha = T(0.7)
    ε     = T(1e-4)
    deg   = 3
    nbp   = 8
    nqp   = 6

    breaks = Bases.hybrid_equidistribution_breakpoints(alpha, ε, nbp, T)
    n      = length(BSplines.BSplineBasis(BSplineOrder(deg + 1), copy(breaks)))
    λ      = complex.(T.(range(1.0, 0.1; length=n)), zeros(T, n))

    InvariantDensityResult{T}(
        alpha, ε, deg, nbp, nqp, decouple,
        breaks,
        Matrix{T}(I, n, n),
        Matrix{T}(I, n, n),
        real.(λ), imag.(λ),
        ones(T, n) ./ n,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Float64 round-trip
# ─────────────────────────────────────────────────────────────────────────────

@testset "Float64 save/load round-trip" begin
    r    = make_result(Float64)
    path = tempname() * ".jld2"
    save_result(path, r)
    r2 = load_result(path)

    @test r2 isa InvariantDensityResult{Float64}
    @test r2.alpha            == r.alpha
    @test r2.epsilon          == r.epsilon
    @test r2.degree           == r.degree
    @test r2.num_break_points == r.num_break_points
    @test r2.num_quad_points  == r.num_quad_points
    @test r2.decouple         == r.decouple
    @test r2.break_points     == r.break_points
    @test r2.M                == r.M
    @test r2.G                == r.G
    @test r2.eigenvalues_re   == r.eigenvalues_re
    @test r2.eigenvalues_im   == r.eigenvalues_im
    @test r2.c                == r.c
end

# ─────────────────────────────────────────────────────────────────────────────
# Float64 load with explicit T
# ─────────────────────────────────────────────────────────────────────────────

@testset "Float64 load with explicit T" begin
    r    = make_result(Float64)
    path = tempname() * ".jld2"
    save_result(path, r)
    r2 = load_result(path, Float64)

    @test r2 isa InvariantDensityResult{Float64}
    @test r2.alpha            == r.alpha
    @test r2.epsilon          == r.epsilon
    @test r2.degree           == r.degree
    @test r2.num_break_points == r.num_break_points
    @test r2.num_quad_points  == r.num_quad_points
    @test r2.decouple         == r.decouple
    @test r2.break_points     == r.break_points
    @test r2.M                == r.M
    @test r2.G                == r.G
    @test r2.eigenvalues_re   == r.eigenvalues_re
    @test r2.eigenvalues_im   == r.eigenvalues_im
    @test r2.c                == r.c
end

# ─────────────────────────────────────────────────────────────────────────────
# Double64 round-trip — bit-exact hi/lo preservation
# ─────────────────────────────────────────────────────────────────────────────

@testset "Double64 save/load round-trip (bit-exact)" begin
    r    = make_result(Double64)
    path = tempname() * ".jld2"
    save_result(path, r)
    r2 = load_result(path)

    @test r2 isa InvariantDensityResult{Double64}

    # == on Double64 checks both hi and lo for scalars and arrays
    @test r2.alpha          == r.alpha
    @test r2.epsilon        == r.epsilon
    @test r2.decouple       == r.decouple
    @test r2.break_points   == r.break_points
    @test r2.c              == r.c
    @test r2.M              == r.M
    @test r2.G              == r.G
    @test r2.eigenvalues_re == r.eigenvalues_re
    @test r2.eigenvalues_im == r.eigenvalues_im
end

# ─────────────────────────────────────────────────────────────────────────────
# Rehydration — Float64
# ─────────────────────────────────────────────────────────────────────────────

@testset "rehydrate Float64" begin
    r  = make_result(Float64)
    rh = rehydrate(r)

    @test rh.basis isa BSplines.BSplineBasis
    @test rh.pdf   isa BSplineKit.Splines.Spline
    @test length(rh.eigenvalues) == length(r.eigenvalues_re)
    @test rh.eigenvalues ≈ complex.(r.eigenvalues_re, r.eigenvalues_im)

    # Spline evaluates at an interior point without error
    x = Float64(0.3)
    @test isfinite(rh.pdf(x))

    # Spline coefficients match what was stored
    @test BSplineKit.coefficients(rh.pdf) == r.c
end

# ─────────────────────────────────────────────────────────────────────────────
# Rehydration — Double64
# ─────────────────────────────────────────────────────────────────────────────

@testset "rehydrate Double64" begin
    r  = make_result(Double64)
    rh = rehydrate(r)

    @test rh.basis isa BSplines.BSplineBasis
    @test rh.pdf   isa BSplineKit.Splines.Spline
    @test length(rh.eigenvalues) == length(r.eigenvalues_re)

    x = Double64(0.3)
    @test isfinite(rh.pdf(x))
    @test BSplineKit.coefficients(rh.pdf) == r.c
end

# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: save → load → rehydrate
# ─────────────────────────────────────────────────────────────────────────────

@testset "save → load → rehydrate pipeline" begin
    r    = make_result(Float64)
    path = tempname() * ".jld2"
    save_result(path, r)

    rh = rehydrate(load_result(path))

    # Spline from loaded result agrees with spline from original at sample points
    xs = Float64[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    rh_orig = rehydrate(r)
    @test all(rh.pdf(x) == rh_orig.pdf(x) for x in xs)
end

# ─────────────────────────────────────────────────────────────────────────────
# decouple flag preserved through save/load
# ─────────────────────────────────────────────────────────────────────────────

@testset "decouple=true preserved through save/load" begin
    r    = make_result(Float64; decouple=true)
    @test r.decouple == true
    path = tempname() * ".jld2"
    save_result(path, r)
    r2 = load_result(path)
    @test r2.decouple == true
end
