# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

#-------------------------------------------
# Tests for BranchInverse.jl
#-------------------------------------------

const TD = Double64

using PmSpectrum.Utils: construct_left_branch_inverse, construct_tabulated_inverse, make_inverse, symmetric_pm

logspaceT(a::TD, b::TD, n::Integer) = exp10.(range(a, b; length=n))

@testset "Symmetric PM Map Inverse Double64 accuracy" begin
    for αf in (0.1, 0.3, 0.5, 0.7, 0.9)
        α = TD(αf)
        t1_inv = construct_left_branch_inverse(α)
        t2_inv = (y::TD) -> one(TD) - t1_inv(one(TD) - y)

        # Stay off exact 0, 1/2, 1
        xL_hi = TD(0.5) - TD(1e-13)
        xL_lo = TD(1e-13)
        # Left-branch x ∈ (0, 1/2): cluster near 0 and 1/2, then clamp
        g = logspaceT(log10(xL_lo), log10(xL_hi), 600)
        xL = sort!([TD(.5) .- g; g])

        # Right-branch x ∈ (1/2, 1)
        xR = xL .+ TD(0.5)

        # Forward y

        yL = symmetric_pm.(xL, α)
        yR = symmetric_pm.(xR, α)

        # Inverse then forward checks
        xL_hat = t1_inv.(yL)
        xR_hat = t2_inv.(yR)
        yL_hat = symmetric_pm.(xL_hat, α)
        yR_hat = symmetric_pm.(xR_hat, α)

        @test all(isapprox.(xL_hat, xL; rtol=1e-31, atol=1e-31))
        @test all(isapprox.(xR_hat, xR; rtol=1e-31, atol=1e-31))
        @test all(isapprox.(yL_hat, yL; rtol=1e-31, atol=1e-31))
        @test all(isapprox.(yR_hat, yR; rtol=1e-31, atol=1e-31))
    end
end


const TF = Float64

logspaceT(a::TF, b::TF, n::Integer) = exp10.(range(a, b; length=n))

@testset "Symmetric PM Map Inverse Float64 accuracy" begin
    for αf in (0.1, 0.3, 0.5, 0.7, 0.9)
        α = TF(αf)
        t1_inv = construct_left_branch_inverse(α)
        t2_inv = (y::TF) -> one(TF) - t1_inv(one(TF) - y)

        # Stay off exact 0, 1/2, 1
        xL_hi = TF(0.5) - TF(1e-13)
        xL_lo = TF(1e-13)
        # Left-branch x ∈ (0, 1/2): cluster near 0 and 1/2, then clamp
        g = logspaceT(log10(xL_lo), log10(xL_hi), 600)
        xL = sort!([TF(.5) .- g; g])

        # Right-branch x ∈ (1/2, 1)
        xR = xL .+ TF(0.5)

        yL = symmetric_pm.(xL, α)
        yR = symmetric_pm.(xR, α)

        xL_hat = t1_inv.(yL)
        xR_hat = t2_inv.(yR)
        yL_hat = symmetric_pm.(xL_hat, α)
        yR_hat = symmetric_pm.(xR_hat, α)

        @test all(isapprox.(xL_hat, xL; rtol=1e-15, atol=1e-15))
        @test all(isapprox.(xR_hat, xR; rtol=1e-15, atol=1e-15))
        @test all(isapprox.(yL_hat, yL; rtol=1e-15, atol=1e-15))
        @test all(isapprox.(yR_hat, yR; rtol=1e-15, atol=1e-15))
    end
end


# Test map: f(x) = x + x²/2 on [0, 1]
#   f′(x) = 1 + x  (strictly positive — no degenerate derivative at endpoints)
#   f″(x) = 1      (constant — clean Halley convergence)
#   true inverse:  y ↦ -1 + √(1 + 2y)
#   image:  [0, 3/2]

@testset "Generic C^2 strictly monotone inverse Float64 accuracy" begin
    _f(x)   = x + x^2/2
    _df(x)  = 1 + x
    _ddf(x) = one(x)

    inv_tab  = construct_tabulated_inverse(0.0, 1.0, _f, _df, _ddf)
    inv_true = y -> -1.0 + sqrt(1.0 + 2y)
    inv_make = make_inverse(_f, 0.0, 1.0; atol=1e-15, rtol=1e-15)

    ys = collect(range(1e-6, _f(1.0) - 1e-6; length=500))

    @test all(isapprox.(inv_tab.(ys), inv_true.(ys); rtol=1e-14, atol=1e-14))
    @test all(isapprox.(inv_tab.(ys), inv_make.(ys); rtol=1e-14, atol=1e-14))
end

@testset "Generic C^2 strictly monotone inverse Double64 accuracy" begin
    _f(x)   = x + x^2/2
    _df(x)  = 1 + x
    _ddf(x) = one(x)

    inv_tab  = construct_tabulated_inverse(TD(0), TD(1), _f, _df, _ddf)
    inv_true = y -> TD(-1) + sqrt(TD(1) + 2y)
    inv_make = make_inverse(_f, TD(0), TD(1); atol=1e-32, rtol=1e-32)

    ys = collect(range(TD(1e-6), TD(_f(1.0)) - TD(1e-6); length=500))

    @test all(isapprox.(inv_tab.(ys), inv_true.(ys); rtol=1e-30, atol=1e-30))
    @test all(isapprox.(inv_tab.(ys), inv_make.(ys); rtol=1e-30, atol=1e-30))
end


@testset "make_inverse: argument validation" begin
    _f(x) = x + x^2/2

    @test_throws ArgumentError make_inverse(_f, 1.0, 0.0)                    # a < b
    @test_throws ArgumentError make_inverse(_f, 0.0, 1.0; method=:newton)    # unknown method

    # An unrecognised `monotone` must not fall through to :auto. :banana would
    # otherwise be silently accepted and behave exactly like the default.
    @test_throws ArgumentError make_inverse(_f, 0.0, 1.0; monotone=:banana)
    @test_throws ArgumentError make_inverse(_f, 0.0, 1.0; monotone=:increasing)

    # The three documented symbols still work; :inc agrees with :auto here.
    ys = collect(range(1e-6, _f(1.0) - 1e-6; length=50))
    for m in (:auto, :inc)
        @test all(isapprox.(make_inverse(_f, 0.0, 1.0; monotone=m).(ys),
                            (-1.0) .+ sqrt.(1.0 .+ 2 .* ys); rtol=1e-10, atol=1e-10))
    end
    @test make_inverse(_f, 0.0, 1.0; monotone=:dec) isa Function

    # y outside f([a,b]) is a DomainError, not an ArgumentError
    @test_throws DomainError make_inverse(_f, 0.0, 1.0)(2.0)
end
