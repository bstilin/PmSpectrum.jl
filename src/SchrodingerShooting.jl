###############################################################################
# Matched shooting solver for the truncated full-line Schrödinger eigenproblem
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
#   α   : exponent appearing in the potential q_α,
#   L   : truncation point used to approximate the half-line [0,∞),
#   x_m : matching point in the interior of (0,L),
#   λ   : trial spectral parameter.
#
# For a trial value of λ, the eigenvalue equation is written as the second-order
# initial-value problem
#
#   u''(x) = (q_α(x) - λ) u(x).
#
# One solution is integrated forward from x=0 using the boundary condition for
# the chosen parity sector, while a second solution is integrated backward from
# x=L. The two solutions correspond to the same eigenfunction exactly when they
# are linearly dependent at the matching point x_m. Equivalently, the Wronskian
# of their phase vectors vanishes there:
#
#   trial λ  →  left IVP + right IVP  →  phase vectors at x_m  →  W  →  F(λ).
#
# The eigenvalues are therefore the roots of the scalar matching function F(λ).
# A bracket around each root may be obtained from the inexpensive
# finite-difference solver and then refined with a one-dimensional root finder.
#
#
# This solver does not choose L, x_m, eigenvalue brackets, or numerical
# tolerances automatically; these are supplied by the caller.
###############################################################################


"""
    _T(args...)

Working float type: the promotion of everything the caller supplied. Used so the
initial data, time span, and default tolerances are all built in one consistent
type and nothing silently narrows to `Float64`.
"""
_T(args...) = float(promote_type(map(typeof, args)...))


"""
    left_initial_data(parity, ::Type{T}) -> (y, dy)

Initial phase vector at `x = 0`, in **mathematical order** `(y, y')`.

`parity` is what selects the **left** boundary condition — it plays the role at
`x = 0` that `right_boundary` plays at `x = L`:

- `:even` — `y'(0) = 0`, so `(1, 0)`.
- `:odd`  — `y(0) = 0`,  so `(0, 1)`.

The unspecified portion of the IVP data is an arbitrary non-zero scale.
"""
function left_initial_data(parity::Symbol, ::Type{T}) where {T<:AbstractFloat}
    parity === :even && return (one(T), zero(T))
    parity === :odd  && return (zero(T), one(T))
    throw(ArgumentError("parity must be :even or :odd, got :$parity"))
end


"""
    right_initial_data(alpha, lambda, L; right_boundary=:dirichlet) -> (y, dy)

Initial phase vector at `x = L` for the backward integration, in mathematical
order `(y, y')`. The slope is an arbitrary non-zero scale.

- `:dirichlet` — the truncated condition `y(L) = 0`, so `(0, 1)`.

The two ends of the interval are selected by different arguments, and
deliberately so: **`parity` fixes the condition at `x = 0`, `right_boundary`
fixes the one at `x = L`.** The parity sector says nothing about the right end —
both sectors truncate the same way — which is why the right condition is a
separate knob rather than another case of `parity`.

`alpha` and `lambda` are accepted but unused by `:dirichlet`. They are in the
signature so a `:wkb` decaying condition can be added here later.
"""
function right_initial_data(alpha, lambda, L; right_boundary::Symbol = :dirichlet)
    T = _T(alpha, lambda, L)
    right_boundary === :dirichlet && return (zero(T), one(T))
    throw(ArgumentError("right_boundary must be :dirichlet (:wkb not implemented), got :$right_boundary"))
end


"""
    phase_state(sol, x) -> (y, dy)

The phase vector in **mathematical order** `(y, y')`.

This exists because `SecondOrderODEProblem` stores the state the other way round.
From the SciMLBase constructor,

    SecondOrderODEProblem{iip}(f, du0, u0, tspan, p) → ArrayPartition((du0, u0))

so `sol(x)[1]` is the *velocity* `y'` and `sol(x)[2]` is the *position* `y` — and
the constructor likewise takes `du0` before `u0`. Confining that reversal to this
one function keeps every other line of the file in the order the mathematics uses.
"""
phase_state(sol, x) = (sol(x)[2], sol(x)[1])


"""
    _solve_side(alpha, lambda, x0, x1, y, dy; ode_alg, abstol, reltol, dtmax)

Integrate `y'' = (q_α - λ) y` from `x0` to `x1`, starting from the phase vector
`(y, dy)` given in mathematical order. Returns the `ODESolution`.

# Arguments
- `alpha`  : exponent α of the potential `q_α`; passed through to `potential`.
- `lambda` : the trial eigenvalue λ. It enters only as the constant shift in
             `y'' = (q_α(x) - λ) y`, so a root find in λ re-solves this IVP but
             never rebuilds anything else.
- `x0`     : where the integration starts, and where `(y, dy)` are imposed.
- `x1`     : where it ends. **Backward integration is simply `x1 < x0`** — the
             right-hand solve runs from `L` down to `xmatch` — and needs no other
             change; the integrator reads the direction off the time span.
- `y`, `dy`: the initial phase vector in mathematical order `(y, y')`. Note these
             are handed to `SecondOrderODEProblem` in the opposite order, which is
             the same reversal `phase_state` undoes on the way out.

# Keyword arguments
- `ode_alg` : the integrator. Anything from `OrdinaryDiffEqRKN` applies, because
  the acceleration ignores its velocity argument and Runge-Kutta-Nyström methods
  require exactly that. Available: `DPRKN4`, `DPRKN5`, `DPRKN6`, `DPRKN6FM`,
  `DPRKN8`, `DPRKN12` (the default, and the highest order), `ERKN4/5/7`,
  `FineRKN4/5`, `Nystrom4`, `Nystrom4VelocityIndependent`,
  `Nystrom5VelocityIndependent`, `RKN4`, `IRKN3`, `IRKN4`. Lowering the order is
  the cheapest way to check that a result is not an artefact of one integrator.

- `abstol`, `reltol` : passed straight to `solve`, where they drive **adaptive
  step-size control** — a step is accepted when its local error estimate satisfies
  `‖err / (abstol + reltol·|u|)‖ ≤ 1`. They therefore control accuracy *at the
  step points*, which is what the eigenvalue depends on. Both default to
  `eps(T)^(3/4)` (1.8e-12 in `Float64`, 3.3e-24 in `Double64`).

- `dtmax` : an upper bound on the step size the integrator may take; `Inf`
  (default) lets it choose freely. Normally unnecessary for endpoint accuracy:
  `abstol`/`reltol` are the primary control, since they bound the local error at
  the step points. It is not inert, though — capping the step changes the step
  sequence and so the accumulated global error — it is simply not the knob to
  reach for. What it does control directly is the accuracy of the interpolant
  when evaluating the solution between step points.

- `q` : the potential, called as `q(x, alpha)`; defaults to `potential`, i.e.
  q_α. Supplying another runs the identical shooting machinery on a different
  problem, which is how the solver is checked against known spectra —
  `q = (x, _) -> zero(x)` for the free particle, `q = (x, _) -> x^2` for the
  harmonic oscillator. A custom `q` may ignore `alpha`, which is still passed.
  Threaded through `shooting_data`, `matching_residual`, `refine_eigenvalue`,
  `shooting_eigenfunction` and `turning_point`, and forwarded automatically by
  `refine_spectrum`.
"""
function _solve_side(alpha, lambda, x0, x1, y, dy;
                     ode_alg, abstol, reltol, dtmax, q = potential)
    T = _T(alpha, lambda, x0, x1, y, dy)

    # y'' = (q(x, α) - λ) y.  The velocity argument is unused, which is what lets
    # Runge-Kutta-Nyström methods apply.
    accel(dyv, yv, p, x) = (q(x, p[1]) - p[2]) .* yv

    # Note the argument order: derivative first, then position (see phase_state).
    prob = SecondOrderODEProblem(accel, T[dy], T[y], (T(x0), T(x1)), (T(alpha), T(lambda)))
    return solve(prob, ode_alg; abstol = abstol, reltol = reltol, dtmax = dtmax)
end


"""
    shooting_data(alpha, lambda, L, xmatch; parity=:even, right_boundary=:dirichlet,
                  ode_alg=DPRKN12(), abstol, reltol, dtmax=Inf,
                  q=potential) -> NamedTuple

Both IVP solves for one trial `lambda`, the phase vectors at `xmatch`, and both
Wronskians. This is the inspection-oriented entry point; the two `ODESolution`s
are returned so the integration itself can be examined.

The eigenvalue condition is linear dependence of the left and right phase vectors
at the matching point.

`q` is the potential, called as `q(x, alpha)`; it defaults to `potential`, i.e.
q_α. Supplying another runs this same machinery on a different problem, which is
how the solver is checked against closed-form spectra. A custom `q` may ignore
`alpha`, which is still passed to it.
"""
function shooting_data(alpha, lambda, L, xmatch;
                       parity::Symbol   = :even,
                       right_boundary::Symbol = :dirichlet,
                       ode_alg          = DPRKN12(),
                       abstol           = eps(_T(alpha, lambda, L, xmatch))^(3//4),
                       reltol           = eps(_T(alpha, lambda, L, xmatch))^(3//4),
                       dtmax            = Inf,
                       q                = potential)

    T = _T(alpha, lambda, L, xmatch)
    zero(T) < xmatch < T(L) ||
        throw(ArgumentError("require 0 < xmatch < L, got xmatch=$xmatch, L=$L"))

    yL0, dyL0 = left_initial_data(parity, T)
    yR0, dyR0 = right_initial_data(alpha, lambda, L; right_boundary = right_boundary)

    # Each side is integrated to xmatch so the phase vectors are read off at an
    # endpoint, never from the interpolant (see `_solve_side`).
    left  = _solve_side(alpha, lambda, zero(T), T(xmatch), yL0, dyL0;
                        ode_alg = ode_alg, abstol = abstol, reltol = reltol, dtmax = dtmax, q = q)
    right = _solve_side(alpha, lambda, T(L), T(xmatch), yR0, dyR0;
                        ode_alg = ode_alg, abstol = abstol, reltol = reltol, dtmax = dtmax, q = q)

    YL = phase_state(left,  T(xmatch))
    YR = phase_state(right, T(xmatch))

    raw = YL[1] * YR[2] - YL[2] * YR[1]

    nL, nR = hypot(YL[1], YL[2]), hypot(YR[1], YR[2])
    normalized = (YL[1] / nL) * (YR[2] / nR) - (YL[2] / nL) * (YR[1] / nR)

    return (alpha = alpha, lambda = lambda, L = L, xmatch = xmatch,
            parity = parity, right_boundary = right_boundary,
            left_solution = left, right_solution = right,
            left_state = YL, right_state = YR,
            raw_wronskian = raw, normalized_wronskian = normalized,
            ode_alg = ode_alg, abstol = abstol, reltol = reltol, dtmax = dtmax, q = q)
end


"""
    matching_residual(alpha, lambda, L, xmatch; normalized=true, ...) -> scalar

The scalar `F(λ)` whose zeros are the eigenvalues: the normalized Wronskian by
default, the raw one when `normalized = false`.

Every keyword is forwarded to `shooting_data`, including `q`, the potential —
`q(x, alpha)`, defaulting to `potential`. 
"""
function matching_residual(alpha, lambda, L, xmatch;
                           parity::Symbol   = :even,
                           right_boundary::Symbol = :dirichlet,
                           normalized::Bool = true,
                           ode_alg          = DPRKN12(),
                           abstol           = eps(_T(alpha, lambda, L, xmatch))^(3//4),
                           reltol           = eps(_T(alpha, lambda, L, xmatch))^(3//4),
                           dtmax            = Inf,
                           q                = potential)

    d = shooting_data(alpha, lambda, L, xmatch; parity = parity, right_boundary = right_boundary,
                      ode_alg = ode_alg, abstol = abstol, reltol = reltol, dtmax = dtmax, q = q)
    return normalized ? d.normalized_wronskian : d.raw_wronskian
end


"""
    refine_eigenvalue(alpha, bracket, L, xmatch; parity=:even, ...) -> NamedTuple

Refine one eigenvalue by finding the zero of `matching_residual` inside the
supplied `bracket = (λ_lo, λ_hi)`.

The bracket is **always** the caller's; nothing here searches for one. `F` is
evaluated at both endpoints first. An endpoint at which `F` is exactly zero is
itself the root and is returned as such; otherwise a bracket that does not
straddle zero is an error reporting both residuals rather than a silent failure.
In the returned-endpoint case the root finder never runs, so `tracks` comes back
empty and `steps` and `fncalls` are zero.

# Arguments
- `alpha::Real` : exponent α of the potential `q_α`, with `0 < α < 1`.
- `bracket::NTuple{2,Real}` : `(λ_lo, λ_hi)` with `λ_lo < λ_hi`, straddling
  exactly one root of `F`.
- `L::Real` : right truncation point of the half-line, **not** the boundary-layer
  scale `L_ε`.
- `xmatch::Real` : matching point for the left and right solutions, `0 < xmatch < L`. 
  `turning_point(alpha, λ)` is the usual choice.

# Keyword arguments — the problem
- `parity::Symbol` : `:even` (`y'(0)=0`) or `:odd` (`y(0)=0`); sets the condition
  at the **left** end.
- `right_boundary::Symbol` : condition at `x = L`; `:dirichlet` only for now.
- `normalized::Bool` : use the normalized Wronskian (default) or the raw one.
  Same roots either way;

# Keyword arguments — the two solvers
`abstol`, `reltol` and `dtmax` are forwarded verbatim to `solve`; the
`root_`-prefixed ones to Roots' `find_zero` as `xatol`/`xrtol`/`maxevals`. See
those packages for what each controls; the prefix exists only to keep the two
sets apart.

The root finder is fixed to `Roots.A42()` and is not a keyword. It converges
here in 4–5 steps, and the answer is set by the ODE tolerance rather than by the
root method, so the choice earns nothing. Fixing it also removes `atol`/`rtol`
entirely: A42 ignores residual tolerances, so they could only ever be dead
weight. To try another method, call `matching_residual` and run your own
`find_zero` over it — the residual is the whole interface.

- `ode_alg` : any Runge-Kutta-Nyström method from `OrdinaryDiffEqRKN`;
  `DPRKN12()` default.
- `abstol`, `reltol::Real` : ODE tolerances, default `eps(T)^(3/4)`. These set how
  accurately each `F(λ)` is computed, and so the eigenvalue's accuracy floor.
- `dtmax::Real` : maximum integrator step, `Inf` default. Normally unnecessary
  here: each side is integrated exactly to `xmatch`, so the residual is read at
  an integration endpoint rather than from the interpolant, and `abstol`/`reltol`
  are the accuracy control. Capping the step still perturbs the accumulated
  integration error — it is just not the knob to reach for.
- `root_xatol`, `root_xrtol::Real` : convergence on the bracket width in λ.
  `root_xrtol` defaults to `eps(T)`, not `0`; with `0`, A42 shrinks the bracket
  past the noise floor of `F` and exhausts its budget without converging, on the
  same answer.
- `root_maxevals::Integer` : iteration cap, default `100`.
- `q` : the potential, called as `q(x, alpha)`; defaults to `potential`, i.e.
  q_α. Swapping it points the whole root find at a different problem, which is
  how the solver is validated against the free particle and the harmonic
  oscillator. A custom `q` may ignore `alpha`. Note `turning_point` takes the
  same keyword, so a custom potential must be given to both.

The returned `tracks` is Roots' own `Tracks` object, so the full iteration
history is available without any custom logging.

This function does **not** decide whether `L`, `xmatch`, or the tolerances are adequate.
"""
function refine_eigenvalue(alpha, bracket, L, xmatch;
                           parity::Symbol   = :even,
                           right_boundary::Symbol = :dirichlet,
                           normalized::Bool = true,
                           ode_alg          = DPRKN12(),
                           abstol           = eps(_T(alpha, first(bracket), L, xmatch))^(3//4),
                           reltol           = eps(_T(alpha, first(bracket), L, xmatch))^(3//4),
                           dtmax            = Inf,
                           root_xatol       = zero(_T(alpha, first(bracket), L, xmatch)),
                           root_xrtol       = eps(_T(alpha, first(bracket), L, xmatch)),
                           root_maxevals    = 100,
                           q                = potential)

    lo, hi = first(bracket), last(bracket)
    lo < hi || throw(ArgumentError("bracket must satisfy lo < hi, got $bracket"))

    F(λ) = matching_residual(alpha, λ, L, xmatch; parity = parity, right_boundary = right_boundary,
                             normalized = normalized, ode_alg = ode_alg,
                             abstol = abstol, reltol = reltol, dtmax = dtmax, q = q)

    Flo, Fhi = F(lo), F(hi)

    # A42 is fixed rather than exposed. It converges here in 4-5 steps, and it
    # ignores residual tolerances entirely, so `atol`/`rtol` need not exist.
    tracks = Tracks()

    # An endpoint that is itself a root is an answer, not a failure — and it has
    # to be caught before the sign test, because signbit(0.0) is false: an exact
    # zero reads as positive and the straddle check would reject the bracket.
    # `iszero` covers -0.0 too. A short-circuited `tracks` stays empty.
    λ = if iszero(Flo)
            lo
        elseif iszero(Fhi)
            hi
        else
            signbit(Flo) == signbit(Fhi) && throw(ArgumentError(
                "bracket ($lo, $hi) does not straddle a sign change of the matching residual: " *
                "F(lo) = $Flo, F(hi) = $Fhi. Widen or relocate the bracket."))

            find_zero(F, (lo, hi), A42(); tracks = tracks,
                      xatol = root_xatol, xrtol = root_xrtol, maxevals = root_maxevals)
        end

    d = shooting_data(alpha, λ, L, xmatch; parity = parity, right_boundary = right_boundary,
                      ode_alg = ode_alg, abstol = abstol, reltol = reltol, dtmax = dtmax, q = q)

    return (eigenvalue = λ,
            residual = normalized ? d.normalized_wronskian : d.raw_wronskian,
            raw_wronskian = d.raw_wronskian,
            normalized_wronskian = d.normalized_wronskian,
            bracket = (lo, hi), bracket_residuals = (Flo, Fhi),
            alpha = alpha, L = L, xmatch = xmatch,
            parity = parity, right_boundary = right_boundary, normalized = normalized,
            ode_alg = ode_alg, abstol = abstol, reltol = reltol, dtmax = dtmax,
            root_xatol = root_xatol, root_xrtol = root_xrtol,
            root_maxevals = root_maxevals,
            tracks = tracks, steps = tracks.steps, fncalls = tracks.fncalls,
            convergence_flag = tracks.convergence_flag)
end


"""
    shooting_eigenfunction(alpha, lambda, L, xmatch; parity=:even, ...) -> NamedTuple

Reconstruct the eigenfunction at an already-refined `lambda`. 

At the matching point the two phase vectors are proportional, `Y_L = c Y_R`. The
constant is taken from whichever component of `Y_R` is larger in absolute value,
so a component that happens to be near zero is never the divisor. The spliced
solution is

    ỹ(x) = y_L(x)      for 0 ≤ x ≤ x_m
           c y_R(x)    for x_m ≤ x ≤ L

then scaled so that `∫₀ᴸ ỹ² = 1`, with `xmatch` passed to `quadgk` as an interior
breakpoint so no quadrature panel straddles the splice.

# Arguments
- `alpha::Real` : exponent α of the potential `q_α`, with `0 < α < 1`.
- `lambda::Real` : the eigenvalue, **already refined** — typically
  `refine_eigenvalue(...).eigenvalue`. Nothing here checks it. At a λ that is not
  an eigenvalue the two solutions are not proportional at `xmatch`, and the splice
  silently produces a function that is not an eigenfunction; the returned
  Wronskians are what to inspect if in doubt.
- `L::Real` : right truncation point of the half-line, **not** the boundary-layer
  scale `L_ε`.
- `xmatch::Real` : matching point, `0 < xmatch < L`, and the splice point of the
  result. Normally the same one used to refine `lambda`.

# Keyword arguments — the problem
- `parity::Symbol` : `:even` (`y'(0)=0`) or `:odd` (`y(0)=0`); the condition at
  the **left** end.
- `right_boundary::Symbol` : condition at `x = L`; `:dirichlet` only for now.

# Keyword arguments — the two quadratures
`abstol`, `reltol` and `dtmax` are forwarded verbatim to `solve`, `quad_rtol` to
`quadgk` as its `rtol`.

- `ode_alg` : any Runge-Kutta-Nyström method from `OrdinaryDiffEqRKN`;
  `DPRKN12()` default.
- `abstol`, `reltol::Real` : ODE tolerances, default `eps(T)^(3/4)`.
- `dtmax::Real` : maximum integrator step, `Inf` default. **Unlike in
  `refine_eigenvalue`, this one matters here** — see the accuracy note below.
- `quad_rtol::Real` : relative tolerance for the `L²` normalisation integral,
  default `eps(T)^(3/4)`. It bounds the accuracy of `normalization_constant`, not
  of the eigenfunction's shape.
- `q` : the potential, called as `q(x, alpha)`; defaults to `potential`. Must be
  the same one `lambda` was refined with, or the two solutions will not match.

Accuracy note: the returned callable evaluates the ODE solutions at arbitrary
points, i.e. through the interpolant packaged with the solver, which is lower order 
than the integrator itself (see `_solve_side`). The eigen*value* is read at an
integration endpoint and so is insensitive to this, but for a pointwise-accurate
eigen*function*, pass a `dtmax` small enough to keep the interpolation error where
you need it. With the default `dtmax = Inf` expect roughly 1e-5 pointwise.
"""
function shooting_eigenfunction(alpha, lambda, L, xmatch;
                                parity::Symbol   = :even,
                                right_boundary::Symbol = :dirichlet,
                                ode_alg          = DPRKN12(),
                                abstol           = eps(_T(alpha, lambda, L, xmatch))^(3//4),
                                reltol           = eps(_T(alpha, lambda, L, xmatch))^(3//4),
                                dtmax            = Inf,
                                quad_rtol        = eps(_T(alpha, lambda, L, xmatch))^(3//4),
                                q                = potential)

    T = _T(alpha, lambda, L, xmatch)
    d = shooting_data(alpha, lambda, L, xmatch; parity = parity, right_boundary = right_boundary,
                      ode_alg = ode_alg, abstol = abstol, reltol = reltol, dtmax = dtmax, q = q)

    YL, YR = d.left_state, d.right_state

    # Y_L = c Y_R; divide by the larger component of Y_R.
    c = abs(YR[1]) ≥ abs(YR[2]) ? YL[1] / YR[1] : YL[2] / YR[2]

    spliced(x) = x ≤ T(xmatch) ? phase_state(d.left_solution,  T(x))[1] :
                             c * phase_state(d.right_solution, T(x))[1]

    sq, _ = quadgk(x -> spliced(x)^2, zero(T), T(xmatch), T(L); rtol = quad_rtol)
    scale = inv(sqrt(sq))

    return (alpha = alpha, lambda = lambda, L = L, xmatch = xmatch,
            parity = parity, right_boundary = right_boundary,
            left_solution = d.left_solution, right_solution = d.right_solution,
            match_scale = c, normalization_constant = scale,
            left_state = YL, right_state = YR,
            raw_wronskian = d.raw_wronskian,
            normalized_wronskian = d.normalized_wronskian,
            ode_alg = ode_alg, abstol = abstol, reltol = reltol, dtmax = dtmax,
            eigenfunction = x -> scale * spliced(x))
end


"""
    turning_point(alpha, lambda; q=potential) -> x

The classical turning point: the unique `x > 0` with `q_α(x) = λ`.

`q_α` is strictly increasing on `(0, ∞)` with `q_α(0) = 0`, so for `λ > 0` the
root exists and is unique. The upper bracket is found by doubling.

`q` is the potential, called as `q(x, alpha)`; it defaults to `potential`. It
must be increasing in `x` for the bracket-by-doubling to terminate, and a
turning point only exists where `q` actually reaches `λ` — there is none for the
free particle, for instance, so `xmatch` must then be chosen by hand.

A convenience for *choosing* a matching point — the shooting routines always take
`xmatch` explicitly, and nothing calls this automatically.
"""
function turning_point(alpha, lambda; q = potential)
    T = _T(alpha, lambda)
    lambda > 0 || throw(ArgumentError("require lambda > 0, got $lambda"))

    b = one(T)
    while q(b, T(alpha)) < T(lambda)
        b *= 2
        b > T(1e6) && throw(ArgumentError("no turning point below x = 1e6 for lambda = $lambda"))
    end
    return find_zero(x -> q(x, T(alpha)) - T(lambda), (zero(T), b), Bisection())
end


"""
    bracket_from_fd(fd_eigenvalues, j) -> (lo, hi)

A candidate bracket for the `j`-th eigenvalue, taken as the midpoints to its
neighbours (and a reflected half-gap at the two ends).

# Arguments
- `fd_eigenvalues::AbstractVector{<:Real}` : eigenvalues **only**
  Normally `solve_fd(alpha, L, N; parity=…).eigenvalues`. It must be

  * **a single parity sector**, not the merged spectrum (see below),
  * sorted ascending (`solve_fd` and `refine_spectrum` both return it that way),
  * of length ≥ 2, since a bracket is built from the gap to a neighbour.

  Despite the name nothing requires these to come from the finite-difference
  solver; any ascending estimates of one sector's spectrum will do.
- `j::Integer` : index into that vector, `1 ≤ j ≤ length`.

# One sector, not both
The bracket must be handed to `refine_eigenvalue` with the **same** `parity` the
eigenvalues came from. 

This function only *proposes*; an explicit hand-chosen bracket is always
acceptable instead.
"""
function bracket_from_fd(fd_eigenvalues, j::Integer)
    λ = fd_eigenvalues
    n = length(λ)
    1 ≤ j ≤ n || throw(BoundsError(λ, j))
    n ≥ 2 || throw(ArgumentError("need at least two eigenvalues to form a bracket"))

    lo = j == 1 ? λ[1] - (λ[2] - λ[1]) / 2 : (λ[j-1] + λ[j]) / 2
    hi = j == n ? λ[n] + (λ[n] - λ[n-1]) / 2 : (λ[j] + λ[j+1]) / 2
    return (lo, hi)
end


"""
    brackets_from_fd(fd_eigenvalues) -> Vector{Tuple}

`bracket_from_fd` for every index at once, for feeding `refine_spectrum`.

`fd_eigenvalues` carries the same requirements as in `bracket_from_fd`: the
ascending eigenvalues of **one parity sector**, and eigenvalues only.

# Geometry of the returned brackets
For a strictly increasing input the brackets **abut**: `hi_j` and `lo_{j+1}` are
both `(λ[j] + λ[j+1])/2`, the same expression and so bit-identical. The result is
a partition of `(lo_1, hi_n)` — disjoint interiors, no gaps, `lo` and `hi` each
strictly increasing.

Given an input accurate to better than half the gap to a neighbouring
eigenvalue, each bracket then holds exactly one eigenvalue. That is a large
margin in practice: finite-difference error is ~1e-6 against gaps of 2–6. This
is the assumption `refine_spectrum` relies on

The partition itself requires the input to be strictly increasing, which is not
checked. A repeated eigenvalue lands exactly *on* a shared endpoint, inside
neither neighbour; an unsorted input produces `lo > hi`, which
`refine_eigenvalue` rejects.
"""
brackets_from_fd(fd_eigenvalues) =
    [bracket_from_fd(fd_eigenvalues, j) for j in eachindex(fd_eigenvalues)]

"""
    refine_spectrum(alpha, brackets, L, xmatches; parity=:even, ...) -> NamedTuple

Refine several eigenvalues of one parity sector. `brackets` and `xmatches` are
explicit vectors of equal length. The matching points are not chosen here; a
typical caller uses

    xmatches = turning_point.(alpha, fd.eigenvalues)

when the brackets are constructed from approximate finite-difference (FD)
eigenvalues.

Each bracket is refined independently by `refine_eigenvalue`; this function is
just the loop, and adds no checking of its own.

# Assumption on the brackets
`brackets` is assumed to be an increasing sequence of disjoint intervals holding
exactly one eigenvalue each — which is what `brackets_from_fd` produces from a
reasonable finite-difference spectrum.

The assumption fails only if the input spectrum is wrong by more than half the
gap to a neighbouring eigenvalue.

All other keywords, `q` (the potential) among them, are forwarded verbatim to
`refine_eigenvalue`.
"""
function refine_spectrum(alpha, brackets, L, xmatches;
                         parity::Symbol = :even, kwargs...)

    length(brackets) == length(xmatches) ||
        throw(ArgumentError("brackets and xmatches must have equal length, got " *
                            "$(length(brackets)) and $(length(xmatches))"))

    results = [refine_eigenvalue(alpha, brackets[j], L, xmatches[j];
                                 parity = parity, kwargs...)
               for j in eachindex(brackets)]

    return (eigenvalues = [r.eigenvalue for r in results], results = results,
            brackets = brackets, xmatches = xmatches, parity = parity)
end


"""
    merge_parity(even_spectrum, odd_spectrum) -> NamedTuple

Merge two `refine_spectrum` results into the full-line spectrum, ordered by
eigenvalue and labelled by parity.

The full-line potential is even, so its spectrum is the union of the two sectors,
and Sturm-Liouville nodal theory says they strictly interleave: even, odd, even, …
`interleaved` reports whether they do. A violation means a mode was missed or is
unresolved — it is reported, not enforced.

`sector_index[j]` is the position of entry `j` within its own sector, so its
eigenfunction can be rebuilt from `results[j]`.
"""
function merge_parity(even_spectrum, odd_spectrum)
    entries = vcat([(λ, :even, j) for (j, λ) in enumerate(even_spectrum.eigenvalues)],
                   [(λ, :odd,  j) for (j, λ) in enumerate(odd_spectrum.eigenvalues)])
    perm = sortperm([e[1] for e in entries])
    entries = entries[perm]

    all_results = vcat(even_spectrum.results, odd_spectrum.results)[perm]
    parities = [e[2] for e in entries]

    interleaved = all(parities[j] === (isodd(j) ? :even : :odd) for j in eachindex(parities))

    return (eigenvalues = [e[1] for e in entries],
            parities = parities,
            sector_index = [e[3] for e in entries],
            results = all_results,
            interleaved = interleaved)
end
