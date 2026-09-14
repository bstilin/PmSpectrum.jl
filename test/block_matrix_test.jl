# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

#-------------------------------------------
# Tests for BlockMatrix.jl:
#   project_global_approximation
#   block decomposition pipeline
#-------------------------------------------

using PmSpectrum
using PmSpectrum.Utils: symmetric_logspace_grid, construct_left_branch_inverse, symmetric_pm
using PmSpectrum.Bases: build_pm_basis, project_global_approximation,
    build_single_splines, mass_matrix, transfer_matrix, Branch,
    EndpointRefinement,
    partition_basis, decompose_matrices, block_spectrum,
    restrict_to_block, embed_block_vector, apply_block
using BSplineKit
using DoubleFloats


######################################################
# Block decomposition smoke test
# (partition → decompose → spectrum → apply on a small decoupled basis)
######################################################

@testset "block decomposition pipeline on a decoupled basis" begin
    T = Float64
    α = T(0.5)
    ε = T(1e-3)

    B, L = build_pm_basis(α, ε, 8, T; decouple=true)
    spls = build_single_splines(B)

    part = partition_basis(spls, L)
    @test part.n == length(spls)
    @test sort(vcat(part.idx_left, part.idx_int, part.idx_right)) == 1:part.n
    @test !isempty(part.idx_left) && !isempty(part.idx_int) && !isempty(part.idx_right)

    left_inv  = construct_left_branch_inverse(α)
    right_inv = x -> one(T) - left_inv(one(T) - x)
    right_fwd = x -> one(T) - symmetric_pm(one(T) - x, α)   # right-branch formula
    b1 = Branch(x -> symmetric_pm(x, α), left_inv,  (zero(T), T(0.5)), (zero(T), one(T)))
    b2 = Branch(right_fwd,               right_inv, (T(0.5),  one(T)), (zero(T), one(T)))

    M = mass_matrix(spls)

    # The pipeline must behave identically with the base and the
    # endpoint-refined S* quadrature partition.
    for refine in (nothing, EndpointRefinement(α))
        G = transfer_matrix(spls, spls, ε, (b1, b2); n_quad=8, refine=refine)

        d = decompose_matrices(G, M, part; atol=T(1e-12))
        @test size(d.G_QQ) == (length(part.idx_int), length(part.idx_int))
        @test d.M_LL == M[part.idx_left, part.idx_left]

        λs, Vs = block_spectrum(d, :int)
        @test length(λs) == length(part.idx_int)
        @test all(isfinite, λs)
        @test issorted(abs.(λs); rev=true)

        # restrict/embed round-trip and one block application
        c = collect(T, 1:part.n)
        @test restrict_to_block(embed_block_vector(restrict_to_block(c, part, :int), part, :int),
                                part, :int) == restrict_to_block(c, part, :int)
        v = apply_block(d, c, :int, :int)
        @test length(v) == part.n
        @test all(iszero, v[part.idx_left]) && all(iszero, v[part.idx_right])
        @test all(isfinite, v)
    end
end


######################################################
# project_global_approximation
######################################################

@testset "project_global_approximation reproduces rho_hat pointwise" begin
    α   = Float64(0.5)
    ε   = Float64(1e-5)
    ε_t = Float64(1e-10)   # small tilde-epsilon for the reference

    # Reference density: invariant density at tiny ε_t ≈ zero-noise limit.
    ref_result = rehydrate(run_single_experiment(α, ε_t, 75, 8; degree=3))
    basis_ref  = ref_result.basis
    coeffs_ref = ref_result.result.c

    # Project onto a fine output basis (same resolution as basis_ref)
    basis_out, _ = build_pm_basis(α, ε, 75, Float64)
    result  = project_global_approximation(α, ε, basis_out, basis_ref, coeffs_ref;
                                           c_match=10, n_quad_smooth=24)
    S_proj  = BSplineKit.Spline(basis_out, result.coeffs)
    rho_hat = result.rho_hat

    # Dense logspace grid including the near-endpoint boundary layer
    xs = symmetric_logspace_grid(2000; decades=-7, include_ends=false, include_mid=false, T=Float64)

    rel_errs = [abs(S_proj(x) - rho_hat(x)) / abs(rho_hat(x)) for x in xs]
    @test maximum(rel_errs) < 1e-3
end
