# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

# -------------------------------------
# Tests for SchrodingerFD.jl
#
# Finite-difference solver for -u'' + q_α u = λ u on (0, L), by parity.
#
#   FD-1  potential formula
#   FD-2  fd_laplacian structure (the endpoint 1 / 3 entries)
#   FD-3  assemble_fd_operator wiring
#   FD-4  free particle: exact spectrum, second order
#   FD-5  harmonic oscillator: exact spectrum, second order
#   FD-6  solve_fd contract (ordering, sizes, normalisation)
#   FD-7  convergence rate on q_α is O(Δx^{1+α}) in the even sector, not O(Δx²)
# -------------------------------------


using LinearAlgebra
using PmSpectrum.SchrodingerFD: potential, fd_laplacian, assemble_fd_operator, solve_fd


######################################################
# The potential
######################################################

@testset "potential: q_α formula and the cusp at 0 (FD-1)" begin
    for α in (0.15, 0.5, 0.85), x in (1e-6, 0.01, 0.3, 1.0, 3.0, 8.0)
        @test potential(x, α) ≈ 2.0^(α - 1) * (1 + α) * x^α +
                                2.0^(2α - 2) * x^(2 + 2α)  rtol=1e-14
    end

    for α in (0.15, 0.5, 0.85)
        @test potential(0.0, α) == 0.0
        # both terms have positive coefficients and positive exponents
        xs = [1e-8, 1e-4, 0.1, 1.0, 4.0, 8.0]
        @test all(potential.(xs, α) .> 0)
        @test issorted(potential.(xs, α))

        # Hölder but not Lipschitz at the origin: q_α(h)/h ~ h^(α-1) diverges.
        # Two decades of h therefore change the quotient by a fixed 100^(1-α) —
        # this is the exponent that caps the FD-7 convergence rate below 2.
        q1 = potential(1e-6, α) / 1e-6
        q2 = potential(1e-8, α) / 1e-8
        @test q2 > q1
        @test q2 / q1 ≈ 100.0^(1 - α) rtol=1e-6
    end
end


######################################################
# Matrix structure
######################################################

@testset "fd_laplacian: endpoint entries and definiteness (FD-2)" begin
    for N in (2, 3, 8, 25)
        Ae = fd_laplacian(N, :even)
        Ao = fd_laplacian(N, :odd)

        @test Ae isa SymTridiagonal
        @test Ao isa SymTridiagonal
        @test length(Ae.dv) == N && length(Ae.ev) == N - 1

        # off-diagonals are all -1
        @test all(Ae.ev .== -1.0)
        @test all(Ao.ev .== -1.0)

        # diagonal: (1,2,…,2,3) even, (3,2,…,2,3) odd.  The leading entry is the
        # whole point: 1 from the Neumann reflection u_0 = u_1, 3 from the
        # Dirichlet reflection u_0 = -u_1.  The trailing 3 is u_{N+1} = -u_N.
        @test Ae.dv[1]   == 1.0
        @test Ao.dv[1]   == 3.0
        @test Ae.dv[end] == 3.0
        @test Ao.dv[end] == 3.0
        N > 2 && @test all(Ae.dv[2:end-1] .== 2.0)
        N > 2 && @test all(Ao.dv[2:end-1] .== 2.0)

        # the Dirichlet end removes the constant mode, so both are SPD
        @test isposdef(Ae)
        @test isposdef(Ao)
    end

    @test_throws ArgumentError fd_laplacian(8, :sideways)
    @test_throws ArgumentError fd_laplacian(1, :even)
end


@testset "assemble_fd_operator: grid and operator wiring (FD-3)" begin
    α, L, N = 0.5, 8.0, 20
    a = assemble_fd_operator(α, L, N; parity=:even)

    @test keys(a) == (:x, :dx, :A, :potential, :operator)
    @test a.dx == L / N
    @test length(a.x) == N
    @test a.x ≈ [(i - 0.5) * a.dx for i in 1:N]
    @test a.x[1] == a.dx / 2                    # midpoint grid: never samples x = 0
    @test a.x[end] ≈ L - a.dx / 2

    @test a.potential ≈ [potential(xi, α) for xi in a.x]

    # H = A/Δx² + diag(V), exactly
    @test a.operator isa SymTridiagonal
    @test a.operator.dv == a.A.dv ./ a.dx^2 .+ a.potential
    @test a.operator.ev == a.A.ev ./ a.dx^2

    @test assemble_fd_operator(α, L, N; parity=:odd).A.dv[1] == 3.0
    @test_throws ArgumentError assemble_fd_operator(α, -1.0, N)
end


######################################################
# Known-answer tests
######################################################

# These go through `solve_fd` itself via its `q` keyword, so they exercise the
# real entry point — assembly, eigensolve and normalisation — rather than a
# hand-built operator. `alpha` is irrelevant once `q` ignores it, so any value
# will do.
_free_fd(x, a) = zero(x)      # free particle
_sq_fd(x, a)   = x^2          # harmonic oscillator

@testset "free particle: exact spectrum and second order (FD-4)" begin
    # -u'' = λu.  even (u'(0)=0, u(L)=0): λ_k = ((k-½)π/L)²
    #             odd  (u(0)=0, u(L)=0): λ_k = (kπ/L)²
    for L in (1.0, 3.0)
        exact(parity, k) = parity === :even ? ((k - 0.5) * π / L)^2 : (k * π / L)^2

        for parity in (:even, :odd)
            λ_c = solve_fd(0.5, L, 400;  parity=parity, nev=3, q=_free_fd).eigenvalues
            λ_f = solve_fd(0.5, L, 1600; parity=parity, nev=3, q=_free_fd).eigenvalues
            for k in 1:3
                ex = exact(parity, k)
                @test λ_c[k] ≈ ex rtol=1e-3
                @test λ_f[k] ≈ ex rtol=1e-4
                # 4× refinement ⇒ ~16× error reduction (q ≡ 0 is smooth)
                ratio = abs(λ_c[k] - ex) / abs(λ_f[k] - ex)
                @test 13.0 < ratio < 19.0
            end
        end
    end
end


@testset "harmonic oscillator: V = x² (FD-5)" begin
    # -v'' + x² v = λ v on the line ⇒ λ = 2k+1;
    # even sector picks out 1,5,9,…, odd sector 3,7,11,…
    L, N = 10.0, 4000
    λe = solve_fd(0.5, L, N; parity=:even, nev=5, q=_sq_fd).eigenvalues
    λo = solve_fd(0.5, L, N; parity=:odd,  nev=5, q=_sq_fd).eigenvalues
    for (k, ex) in enumerate((1, 5, 9, 13, 17))
        @test λe[k] ≈ ex rtol=1e-4
    end
    for (k, ex) in enumerate((3, 7, 11, 15, 19))
        @test λo[k] ≈ ex rtol=1e-4
    end
end


######################################################
# Solver contract
######################################################

@testset "solve_fd: ordering, sizes, normalisation (FD-6)" begin
    α, L, N, nev = 0.5, 8.0, 2000, 5

    for parity in (:even, :odd)
        s = solve_fd(α, L, N; parity=parity, nev=nev)

        @test s.alpha == α && s.L == L && s.N == N && s.parity == parity
        @test s.dx == L / N
        @test length(s.eigenvalues) == nev
        @test size(s.eigenfunctions) == (N, nev)
        @test issorted(s.eigenvalues)
        @test all(s.eigenvalues .> 0)

        # Δx Σ|uᵢ|² = 1
        for k in 1:nev
            @test s.dx * sum(abs2, s.eigenfunctions[:, k]) ≈ 1.0 rtol=1e-13
        end

        # eigenpairs really do solve the discrete problem
        for k in 1:nev
            u = s.eigenfunctions[:, k]
            @test norm(s.operator * u - s.eigenvalues[k] * u) < 1e-8 * norm(s.operator * u)
        end
    end

    # the two sectors interleave (full-line nodal structure)
    e = solve_fd(α, L, N; parity=:even, nev=3).eigenvalues
    o = solve_fd(α, L, N; parity=:odd,  nev=3).eigenvalues
    @test e[1] < o[1] < e[2] < o[2] < e[3] < o[3]

    @test_throws ArgumentError solve_fd(α, L, N; nev=0)
    @test_throws ArgumentError solve_fd(α, L, 10; nev=11)
end
