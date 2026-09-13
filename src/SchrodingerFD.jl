###############################################################################
# Finite-difference solver for the truncated full-line Schrödinger eigenproblem
#
# We consider the eigenvalue problem
#
#   -u''(x) + q_α(x) u(x) = λ u(x),        x ∈ ℝ,
#
# with
#
#   q_α(x) = 2^(α-1) (1+α) |x|^α + 2^(2α-2) |x|^(2+2α),
#
# where 0 < α < 1. The potential is even and grows as |x| → ∞, so the full-line
# problem splits into even and odd parity sectors. We therefore solve only on
# the truncated half-line [0,L] and enforce the boundary conditions:
#
#   even sector :  u'(0) = 0,  u(L) = 0,
#   odd  sector :  u(0)  = 0,  u(L) = 0.
#
# Here:
#
#   α : exponent appearing in the potential q_α,
#   L : truncation point used to approximate the half-line [0,∞),
#   N : number of grid unknowns on [0,L],
#   Δx = L/N : grid spacing.
#
# The discretisation uses the uniform midpoint grid
#
#   x_i = (i - 1/2) Δx,        i = 1,…,N.
#
# The second derivative is approximated with the standard three-point stencil,
# while ghost points enforce the parity condition at x=0 and the Dirichlet
# condition at x=L. This produces a real symmetric tridiagonal matrix, positive
# definite for q_α and for any other nonnegative potential (see `solve_fd`).
#
# The finite-difference solver is intended primarily as a simple and inexpensive
# way to approximate the low-lying spectrum and generate eigenvalue brackets or
# initial guesses for the more accurate shooting solver.
###############################################################################


"""
    potential(x, alpha)

Evaluate the Schrödinger potential q_α(x) for x ≥ 0,

    q_α(x) = 2^(α-1) (1+α) x^α + 2^(2α-2) x^(2+2α),

Note that q_α is only Hölder continuous at the origin: x^α has an unbounded
derivative there. This limits the convergence rate of the finite-difference
scheme.
"""
potential(x, alpha) = 2.0^(alpha - 1) * (1 + alpha) * x^alpha +
                      2.0^(2alpha - 2) * x^(2 + 2alpha)


"""
    fd_laplacian(N, parity) -> SymTridiagonal

Construct the dimensionless discrete operator for `-u''` on a midpoint grid
with `N` unknowns, for either the even or odd parity sector.

Interior rows are the usual `-u_{i-1} + 2u_i - u_{i+1}`. The two end rows differ
from that because the boundary conditions are imposed by ghost points.-

  * `parity = :even`, `u'(0) = 0`.  Reflection gives `u_0 = u_1`, so
    `-u_0 + 2u_1` collapses to `u_1`, hence the leading **1**.
  * `parity = :odd`, `u(0) = 0`.  Then `(u_0 + u_1)/2 = 0`, i.e. `u_0 = -u_1`,
    so `-u_0 + 2u_1` becomes `3u_1`, hence the leading **3**.

So `A_even = tridiag(-1, (1,2,…,2,3), -1)` and `A_odd = tridiag(-1, (3,2,…,2,3), -1)`.
Both are symmetric tridiagonal, positive definite.
"""
function fd_laplacian(N::Integer, parity::Symbol)
    N ≥ 2 || throw(ArgumentError("Require N ≥ 2, got N=$N"))
    parity === :even || parity === :odd ||
        throw(ArgumentError("parity must be :even or :odd, got :$parity"))

    d = fill(2.0, N)
    d[1]   = parity === :even ? 1.0 : 3.0   # u_0 = u_1 (Neumann) or u_0 = -u_1 (Dirichlet)
    d[end] = 3.0                            # u_{N+1} = -u_N  (Dirichlet at x = L)

    return SymTridiagonal(d, fill(-1.0, N - 1))
end


"""
    assemble_fd_operator(alpha, L, N; parity=:even, q=potential) -> NamedTuple

Assemble the finite-difference approximation of

    H = -d²/dx² + q_α(x)

on the truncated half-line [0,L].

Arguments:
- `alpha`: exponent α in the potential q_α, with 0 < α < 1.
- `L`: right truncation point for the half-line [0,∞).
- `N`: number of midpoint-grid unknowns, giving `Δx = L/N`.
- `parity`: `:even` for u'(0)=0 or `:odd` for u(0)=0.
- `q`: the potential, called as `q(x, alpha)`. Defaults to `potential`, i.e. q_α.
  Supplying another lets the same discretisation be checked against a problem
  with a known spectrum — `q = (x, _) -> zero(x)` for the free particle,
  `q = (x, _) -> x^2` for the harmonic oscillator. `alpha` is then unused by the
  potential but still passed, so a custom `q` may simply ignore it.

The resulting discrete operator is

    H = A / Δx² + diag(q_α(x_1), …, q_α(x_N))

on the midpoint grid `x_i = (i-½)Δx` where A comes from fd_laplacian().
...
"""
function assemble_fd_operator(alpha, L, N::Integer; parity::Symbol = :even,
                              q = potential)
    L > 0 || throw(ArgumentError("Require L > 0, got L=$L"))

    dx = L / N
    x  = [(i - 0.5) * dx for i in 1:N]
    A  = fd_laplacian(N, parity)          # also validates N and parity
    V  = [q(xi, alpha) for xi in x]

    H = SymTridiagonal(A.dv ./ dx^2 .+ V, A.ev ./ dx^2)

    return (x = x, dx = dx, A = A, potential = V, operator = H)
end


"""
    solve_fd(alpha, L, N; parity=:even, nev=5, q=potential) -> NamedTuple

Compute the lowest `nev` eigenpairs of the finite-difference Schrödinger
operator on the truncated half-line [0,L], within one parity sector.

Arguments:
- `alpha`: exponent α in the potential q_α, with 0 < α < 1.
- `L`: right truncation point used to approximate [0,∞).
- `N`: number of midpoint-grid unknowns; `dx = L/N`.
- `parity`: `:even` or `:odd`.
- `nev`: number of eigenpairs to return, starting from the smallest eigenvalue.
- `q`: the potential, called as `q(x, alpha)`; defaults to `potential`. See
  `assemble_fd_operator`.

The operator `H` returned by `assemble_fd_operator` is symmetric and tridiagonal,
so its eigenvalues are real and simple, and they are returned in increasing
order. With the default `q = q_α` it is also positive definite — `A` is positive
definite on its own and `q_α ≥ 0` on `x ≥ 0` — giving

    0 < λ₁ < λ₂ < ⋯ < λ_N.

That last guarantee is the potential's, not the discretisation's: it holds for
any `q` sampling nonnegative on the grid, and a sufficiently negative custom `q`
will push eigenvalues to or below zero. Nothing here checks for that.

This function returns `(λ₁, …, λ_nev)` and their corresponding eigenfunctions
for the selected parity sector. Eigenvalues are returned in increasing order.
Eigenfunctions are normalised to approximate unit `L²(0,L)` norm,

    Δx Σᵢ |uᵢ|² = 1.

Returns `(alpha, L, N, parity, dx, x, A, potential, operator, eigenvalues,
eigenfunctions)`.

`Float64` only: the range-restricted symmetric tridiagonal eigensolver is LAPACK.
"""
function solve_fd(alpha, L, N::Integer; parity::Symbol = :even, nev::Integer = 5,
                  q = potential)
    1 ≤ nev ≤ N || throw(ArgumentError("Require 1 ≤ nev ≤ N, got nev=$nev, N=$N"))

    a = assemble_fd_operator(alpha, L, N; parity=parity, q=q)

    # Range-restricted solve: only the lowest `nev` pairs, already ascending.
    F = eigen(a.operator, 1:nev)

    # LAPACK gives unit 2-norm columns; rescale to Δx Σ|uᵢ|² = 1.
    eigenfunctions = F.vectors ./ sqrt(a.dx)

    return (
        alpha          = alpha,
        L              = L,
        N              = N,
        parity         = parity,
        dx             = a.dx,
        x              = a.x,
        A              = a.A,
        potential      = a.potential,
        operator       = a.operator,
        eigenvalues    = F.values,
        eigenfunctions = eigenfunctions,
    )
end
