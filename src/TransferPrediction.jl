###############################################################################
# From Sturm-Liouville eigenpairs to predicted transfer-operator modes
#
# The Sturm-Liouville problem  -v'' + q_α v = λ v  was obtained from the
# eigenvalue problem for the rescaled Fokker-Planck operator A by the change of
# dependent variable
#
#     Φ(y) = e^{Ψ(y)} v(y),      Ψ(y) = 2^(α-1) |y|^(2+α) / (2+α),
#
# which removes the first-derivative term and makes the operator self-adjoint.
# This module undoes that change, translating an eigenpair (λ_j, v_j) into the
# predicted spectrum and eigenfunctions of the noisy transfer operator:
#
#     Λ_{ε,j} ≈ exp(-λ_j L_ε^α)                      transfer eigenvalue
#     σ_{ε,j} = -λ_j L_ε^α                           generator eigenvalue
#     φ_{ε,j}(x) = L_ε^{-1} e^{Ψ(x/L_ε)} v_j(x/L_ε)  eigenfunction in x
#
# and, as L_ε → 0,   1 - Λ_{ε,j} ~ λ_j L_ε^α.
#
# Everything here is solver-agnostic: eigenpairs come in as plain values, so the
# same code serves the finite-difference and shooting solvers, and any
# later one. Nothing in this file solves anything.
#
# NAMING. Four eigenvalue-like quantities are in play and the rest of the
# package names them inconsistently. Throughout this file:
#
#     λ (`lambda`)   Sturm-Liouville eigenvalue
#     Λ (`Lambda`)   transfer-operator eigenvalue, j-th largest NONSTATIONARY
#     L_ε (`L_eps`)  boundary-layer scale        -- NOT the truncation half-width
#     L              truncation half-width       -- never appears here
#
# `scripts/schrodinger_vs_transfer.jl` calls the Sturm-Liouville eigenvalues ω_j,
# a third name for λ_j, and the stored sweep CSVs call the *transfer* eigenvalues
# `lambda_2 … lambda_7`. Column `lambda_{j+2}` pairs with λ_j.
###############################################################################


###############################################################################
# The ε conventions
###############################################################################

"""
    layer_scale(alpha, eps) -> L_ε

Boundary-layer scale in the paper's convention, with `eps` the noise **standard
deviation**:

    L_ε = c ε^(2/(2+α)),      c = (1/2)^(1/(2+α)).

**This is not the same argument as `Utils.boundary_layer_scale`**, which takes the
noise *half-width* `ε_hw` and computes `(ε_hw²/6)^(1/(2+α))`. The two produce the
same number — for a uniform kick on `[-ε_hw, ε_hw]` the standard deviation is
`ε = ε_hw/√3`, and `ε²/2 = ε_hw²/6` identically — but they take different inputs.

Passing a half-width here silently returns an `L_ε` too large by `3^(1/(2+α))`
(about 1.5 at α = 0.5). Convert first:

    L_eps = layer_scale(alpha, sd_from_halfwidth(eps_halfwidth))

The distinct name is deliberate; giving the two functions the same name is
exactly how the √3 gets lost.
"""
function layer_scale(alpha, eps)
    T = float(promote_type(typeof(alpha), typeof(eps)))
    p = inv(T(2) + T(alpha))
    return (one(T) / T(2))^p * T(eps)^(T(2) * p)
end


"""
    sd_from_halfwidth(eps_hw) -> eps

Noise standard deviation from half-width, `ε = ε_hw/√3`, for a uniform kick on
`[-ε_hw, ε_hw]`.

Use this on any ε coming from the transfer-operator side of the package —
`run_single_experiment`, the `epsilon_halfwidth` column of the stored sweeps, or
anything fed to `Utils.boundary_layer_scale` — before passing it to
`layer_scale`.
"""
sd_from_halfwidth(eps_hw) = eps_hw / sqrt(oftype(float(eps_hw), 3))


"""
    halfwidth_from_sd(eps) -> eps_hw

Noise half-width from standard deviation, `ε_hw = √3 ε`. The inverse of
`sd_from_halfwidth`; use it when handing the paper's ε to
`run_single_experiment` or `Utils.boundary_layer_scale`.
"""
halfwidth_from_sd(eps) = sqrt(oftype(float(eps), 3)) * eps


###############################################################################
# The symmetrising weight
###############################################################################

"""
    psi_weight(y, alpha) -> Ψ(y)

The exponent of the symmetrising weight,

    Ψ(y) = 2^(α-1) |y|^(2+α) / (2+α),

defined by `Ψ(y) = ½∫₀^y b(s) ds`, which is what turns the rescaled
Fokker-Planck operator into the self-adjoint Sturm-Liouville form via
`Φ = e^{Ψ} v`.

Returns the **exponent**, not `e^{±Ψ}`, so that the direction is explicit at
every call site: `e^{+Ψ}` reconstructs a transfer mode from a Sturm-Liouville
eigenfunction, `e^{-Ψ}` goes the other way.
"""
function psi_weight(y, alpha)
    T = float(promote_type(typeof(y), typeof(alpha)))
    a = T(alpha)
    return T(2)^(a - one(T)) * abs(T(y))^(T(2) + a) / (T(2) + a)
end


"""
    psi_cutoff(alpha, tol) -> y

Return the radius `|y|` beyond which the reconstruction

    Φ(y) = exp(Ψ(y)) * v(y)

is no longer numerically trustworthy.

The transformed solution `v` is exponentially small:

    v(y) = exp(-Ψ(y)) * Φ(y),

where `Φ(y)` varies only algebraically. Solving for `v` is numerically
convenient, but far enough into the tail the factor `exp(-Ψ(y))` makes `v`
so small that it reaches the solver's **ABSOLUTE error floor**.

Ignoring the slower algebraic prefactor, this loss of information occurs
roughly when

    exp(-Ψ(y)) ≈ tol.

Beyond this point, the computed `v` is dominated by numerical error. Multiplying
by the large inverse factor `exp(Ψ(y))` then amplifies that error back to O(1)
when reconstructing `Φ`.

The cutoff is therefore estimated from

    Ψ(y) = -log(tol).

This is an accuracy limit, not an overflow limit: `exp(Ψ(y))` may still be
perfectly representable long after the computed `v(y)` has become too small
to contain useful information.

For example, at `α = 0.5` and `tol = 1e-15`, the cutoff is `|y| ≈ 6.8`,
whereas overflow of `exp(Ψ)` would occur much farther out.

This function does not clamp or modify any data; it only returns a radius that
callers may use to restrict comparison, fitting, or plotting ranges.
"""
function psi_cutoff(alpha, tol)
    T = float(promote_type(typeof(alpha), typeof(tol)))
    zero(T) < T(tol) < one(T) || throw(ArgumentError("require 0 < tol < 1, got $tol"))
    a = T(alpha)
    return (-log(T(tol)) * (T(2) + a) / T(2)^(a - one(T)))^inv(T(2) + a)
end


###############################################################################
# Parity extension
###############################################################################

"""
    extend_parity(v_half, parity) -> Function

Extend a half-line eigenfunction to the full line:

    :even  →  y ↦ v_half(|y|)
    :odd   →  y ↦ sign(y) · v_half(|y|)

The full-line Sturm-Liouville problem has an even potential, so its spectrum
splits into parity sectors and each is solved on `[0, L]`. The prediction
formulas need the full-line `v_j`.

Both `SchrodingerFD.solve_fd` and `SchrodingerShooting.shooting_eigenfunction`
return half-line objects, so both need this.

`sign(0) == 0` in Julia, so an odd extension is exactly zero at the origin —
which is the odd sector's boundary condition, not an accident.
"""
function extend_parity(v_half, parity::Symbol)
    parity === :even && return y -> v_half(abs(y))
    parity === :odd  && return y -> sign(y) * v_half(abs(y))
    throw(ArgumentError("parity must be :even or :odd, got :$parity"))
end


###############################################################################
# Eigenvalues: Sturm-Liouville ↔ transfer operator
###############################################################################

"""
    predicted_eigenvalue(lambda, alpha, L_eps; form=:exponential) -> Λ

Transfer-operator eigenvalue Λ predicted from a Sturm-Liouville eigenvalue λ:

    :exponential   Λ = exp(-λ L_ε^α)      (default)
    :linear        Λ = 1 - λ L_ε^α        (its linearisation)

`L_eps` is the boundary-layer scale, not the Sturm-Liouville truncation half-width.
"""
function predicted_eigenvalue(lambda, alpha, L_eps; form::Symbol = :exponential)
    T = float(promote_type(typeof(lambda), typeof(alpha), typeof(L_eps)))
    s = T(lambda) * T(L_eps)^T(alpha)          # = -σ, the decay per unit time
    form === :exponential && return exp(-s)
    form === :linear      && return one(T) - s
    throw(ArgumentError("form must be :exponential or :linear, got :$form"))
end

"""
    predicted_eigenvalues(lambdas, alpha, L_eps; form=:exponential) -> Vector

`predicted_eigenvalue` over a whole Sturm-Liouville spectrum. Since the
map is strictly decreasing in `λ`, increasing `λ_j` give decreasing `Λ_{ε,j}`.
"""
predicted_eigenvalues(lambdas, alpha, L_eps; form::Symbol = :exponential) =
    [predicted_eigenvalue(λ, alpha, L_eps; form = form) for λ in lambdas]


"""
    recovered_eigenvalue(Lambda, alpha, L_eps; form=:exponential) -> λ̂

The inverse map: the Sturm-Liouville eigenvalue λ implied by a **measured**
transfer-operator eigenvalue Λ,

    :exponential   λ̂ = -log(Λ) / L_ε^α     (default)
    :linear        λ̂ = (1 - Λ) / L_ε^α
"""
function recovered_eigenvalue(Lambda, alpha, L_eps; form::Symbol = :exponential)
    T = float(promote_type(typeof(Lambda), typeof(alpha), typeof(L_eps)))
    d = T(L_eps)^T(alpha)
    form === :exponential && return -log(T(Lambda)) / d
    form === :linear      && return (one(T) - T(Lambda)) / d
    throw(ArgumentError("form must be :exponential or :linear, got :$form"))
end

"""
    recovered_eigenvalues(Lambdas, alpha, L_eps; form=:exponential) -> Vector

`recovered_eigenvalue` over a vector of measured transfer eigenvalues.
"""
recovered_eigenvalues(Lambdas, alpha, L_eps; form::Symbol = :exponential) =
    [recovered_eigenvalue(Λ, alpha, L_eps; form = form) for Λ in Lambdas]


###############################################################################
# Eigenfunctions
###############################################################################

"""
    predicted_eigenfunction(v, alpha, L_eps; variable=:outer) -> Function

Generate a predicted eigenfunction from a Sturm-Liouville eigenfunction `v`. `v` must be the
**full-line** Sturm-Liouville eigenfunction — see `extend_parity` for the
half-line solvers.

Three related objects, differing only by prefactor and argument scaling:

| `variable`  | formula                                            | is the eigenfunction of |
|-------------|----------------------------------------------------|-------------------------|
| `:rescaled` | `Φ(y) = e^{Ψ(y)} v(y)`                             | the amplitude-rescaled `U_ε` |
| `:inner`    | `φ̃(y) = L_ε⁻¹ e^{Ψ(y)} v(y)`                       | `u_ε`, on the inner scale |
| `:outer`    | `φ(x) = L_ε⁻¹ e^{Ψ(x/L_ε)} v(x/L_ε)`  (default)    | `u_ε`, in the physical variable |

The `L_ε⁻¹` is kept rather than absorbed into the arbitrary eigenfunction scale.
Apply your own normalization as needed.

Only trustworthy for `|y| ≲ psi_cutoff(alpha, tol)`, where `tol` is the absolute
numerical error scale in the exponentially small tail of `v` — for a shooting
solve, its `abstol` — and **not** a relative accuracy. The distinction is the
whole cutoff argument: a relative error held at `tol` into the tail would be
preserved by the reconstruction and there would be no cutoff at all. The cutoff
exists because `v` bottoms out at a fixed absolute floor while `e^{Ψ}` keeps
growing, so past that radius the amplified noise is O(1).

For `variable = :outer` the returned callable takes `x = y·L_ε`, so its argument
limit is `L_ε · psi_cutoff(alpha, tol)`; the `:rescaled` and `:inner` callables
take `y` directly. See `psi_cutoff`.
"""
function predicted_eigenfunction(v, alpha, L_eps; variable::Symbol = :outer)
    variable === :rescaled && return y -> exp(psi_weight(y, alpha)) * v(y)
    variable === :inner    && return y -> exp(psi_weight(y, alpha)) * v(y) / L_eps
    variable === :outer    && return function (x)
        y = x / L_eps
        return exp(psi_weight(y, alpha)) * v(y) / L_eps
    end
    throw(ArgumentError("variable must be :rescaled, :inner or :outer, got :$variable"))
end


###############################################################################
# Bundled entry points
###############################################################################

"""
    predict_transfer_mode(lambda, v, alpha, L_eps; variable=:outer, form=:exponential)

Translate one Sturm-Liouville eigenpair into the corresponding predicted
transfer-operator mode.

Arguments
---------
- `lambda` : Sturm-Liouville eigenvalue `λ_j`
- `v`      : Sturm-Liouville eigenfunction `v_j`, **full line** (see `extend_parity`)
- `alpha`  : PM exponent
- `L_eps`  : boundary-layer scale (see `layer_scale`) — *not* the truncation half-width

Keyword arguments
-----------------
- `variable` : `:rescaled`, `:inner` or `:outer` (default); which of the three
  eigenfunctions to return. See `predicted_eigenfunction`.
- `form`     : `:exponential` (default) or `:linear`. See `predicted_eigenvalue`.

Returns a NamedTuple whose eigenvalue fields are named for which operator they
belong to, because an unqualified `eigenvalue` next to a Sturm-Liouville one is
precisely the confusion this module exists to prevent:

- `sl_eigenvalue`        : `λ_j`, echoed so the pairing is never lost
- `transfer_eigenvalue`  : `Λ_{ε,j}`, the prediction
- `generator_eigenvalue` : `σ_{ε,j} = -λ_j L_ε^α`
- `gap`                  : `1 - Λ_{ε,j}`
- `linearized_gap`       : `λ_j L_ε^α`, the `L_ε → 0` form of `gap`
- `eigenfunction`        : callable, in chosen `variable`

followed by `alpha`, `L_eps`, `variable` and `form`, echoed so the settings
travel with the result.

No trust radius for `eigenfunction` is bundled here: it is
`psi_cutoff(alpha, tol)` for the absolute error floor `tol` of whichever solver
produced `v`, which this function has no way to know. Call `psi_cutoff` with
your own solver's `abstol`.
"""
function predict_transfer_mode(lambda, v, alpha, L_eps;
                               variable::Symbol = :outer,
                               form::Symbol     = :exponential)
    T = float(promote_type(typeof(lambda), typeof(alpha), typeof(L_eps)))
    d = T(L_eps)^T(alpha)
    Λ = predicted_eigenvalue(lambda, alpha, L_eps; form = form)

    return (sl_eigenvalue        = lambda,
            transfer_eigenvalue  = Λ,
            generator_eigenvalue = -T(lambda) * d,
            gap                  = one(T) - Λ,
            linearized_gap       = T(lambda) * d,
            eigenfunction        = predicted_eigenfunction(v, alpha, L_eps; variable = variable),
            alpha = alpha, L_eps = L_eps, variable = variable, form = form)
end


"""
    predict_transfer_spectrum(lambdas, vs, alpha, L_eps; variable=:outer, form=:exponential)

`predict_transfer_mode` over a whole spectrum. `lambdas` and `vs` must
have equal length and be in the same order — typically the full-line merged
ordering from `SchrodingerShooting.merge_parity`, in which case `vs` must be the
matching parity-extended eigenfunctions.

Returns the vectors plus the per-mode NamedTuples:

    (sl_eigenvalues, transfer_eigenvalues, generator_eigenvalues,
     gaps, linearized_gaps, eigenfunctions, modes, alpha, L_eps, form)

Since `λ_j` increase, the `transfer_eigenvalues` decrease — the ordering reverses,
so the smallest Sturm-Liouville eigenvalue predicts the transfer eigenvalue
closest to 1 (excluding the invariant density).
"""
function predict_transfer_spectrum(lambdas, vs, alpha, L_eps;
                                   variable::Symbol = :outer,
                                   form::Symbol     = :exponential)
    length(lambdas) == length(vs) ||
        throw(ArgumentError("lambdas and vs must have equal length, got " *
                            "$(length(lambdas)) and $(length(vs))"))

    modes = [predict_transfer_mode(lambdas[j], vs[j], alpha, L_eps;
                                   variable = variable, form = form)
             for j in eachindex(lambdas)]

    return (sl_eigenvalues       = [m.sl_eigenvalue        for m in modes],
            transfer_eigenvalues = [m.transfer_eigenvalue  for m in modes],
            generator_eigenvalues = [m.generator_eigenvalue for m in modes],
            gaps                 = [m.gap                  for m in modes],
            linearized_gaps      = [m.linearized_gap       for m in modes],
            eigenfunctions       = [m.eigenfunction        for m in modes],
            modes = modes, alpha = alpha, L_eps = L_eps,
            variable = variable, form = form)
end
