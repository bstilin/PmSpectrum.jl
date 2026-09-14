# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

using JLD2
using BSplineKit

# ─────────────────────────────────────────────────────────────────────────────
# In-memory container
# ─────────────────────────────────────────────────────────────────────────────

"""
    InvariantDensityResult{T<:AbstractFloat}

All outputs from a single invariant-density run. Every field is a plain scalar,
integer, or array — no BSplineKit objects — so the struct is trivially
serializable and version-stable.

**Supported types:** `T ∈ {Float64, Double64}`. See module docstring for details.

Fields
------
- `alpha`            : PM map parameter α
- `epsilon`          : noise half-width ε
- `degree`           : B-spline degree
- `num_break_points` : number of breakpoints in the region [x_c, 1/2], where
                       x_c is the crossover point of the hybrid equidistribution
                       mesh. The full knot vector is symmetric about 1/2, so the
                       total number of breakpoints is approximately 2·num_break_points.
- `num_quad_points`  : Gauss-Legendre nodes per subinterval of S*
- `decouple`         : if `true`, the basis was built so that no B-spline support
                       straddles L or 1-L; each spline is entirely contained in
                       [L, 1-L] or its complement
- `break_points`     : breakpoints on [0,1] (unique knot locations, no endpoint multiplicities)
- `M`                : B-spline mass matrix  (n × n)
- `G`                : transfer-operator Galerkin matrix  (n × n)
- `eigenvalues_re`   : real parts of all eigenvalues, sorted |λ| descending
- `eigenvalues_im`   : imaginary parts of all eigenvalues, same order
- `c`                :  B-spline coefficients for the unit mass invariant density
"""
struct InvariantDensityResult{T<:AbstractFloat}
    alpha            :: T
    epsilon          :: T
    degree           :: Int
    num_break_points :: Int
    num_quad_points  :: Int
    decouple         :: Bool
    break_points     :: Vector{T}
    M                :: Matrix{T}
    G                :: Matrix{T}
    eigenvalues_re   :: Vector{T}
    eigenvalues_im   :: Vector{T}
    c                :: Vector{T}
end


# ─────────────────────────────────────────────────────────────────────────────
# Float64 ↔ T lossless decomposition helpers
# ─────────────────────────────────────────────────────────────────────────────

_scalar_to_f64(x::Float64)  = (x,)
_scalar_to_f64(x::Double64) = (x.hi, x.lo)

_scalar_from_f64(v::NTuple{1,Float64}, ::Type{Float64})  = v[1]
_scalar_from_f64(v::NTuple{2,Float64}, ::Type{Double64}) = Double64(v[1], v[2])

_arr_to_f64(a::AbstractArray{Float64})  = (a,)
_arr_to_f64(a::AbstractArray{Double64}) = (getfield.(a, :hi), getfield.(a, :lo))

_arr_from_f64(hi::AbstractArray{Float64}, ::Type{Float64})  = hi
_arr_from_f64(hi::AbstractArray{Float64}, lo::AbstractArray{Float64}, ::Type{Double64}) =
    Double64.(hi, lo)


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

"""
    save_result(path, result)

Write an `InvariantDensityResult` to a JLD2 file at `path`.

All numeric values are decomposed to plain `Float64` before writing:
`Float64` fields are stored as-is; `Double64` fields are split into
`_hi`/`_lo` Float64 pairs. The file therefore has no dependency on
DoubleFloats.jl and is immune to its internal representation changes.

Only `Float64` and `Double64` are supported.
"""
function save_result(path::AbstractString, r::InvariantDensityResult{T}) where {T}
    jldopen(path, "w") do f
        f["type"]             = string(T)
        f["degree"]           = r.degree
        f["num_break_points"] = r.num_break_points
        f["num_quad_points"]  = r.num_quad_points
        f["decouple"]         = r.decouple

        for (name, val) in (("alpha", r.alpha), ("epsilon", r.epsilon))
            parts = _scalar_to_f64(val)
            if length(parts) == 1
                f[name] = parts[1]
            else
                f["$(name)_hi"] = parts[1]
                f["$(name)_lo"] = parts[2]
            end
        end

        for (name, val) in (
            ("break_points",   r.break_points),
            ("M",              r.M),
            ("G",              r.G),
            ("eigenvalues_re", r.eigenvalues_re),
            ("eigenvalues_im", r.eigenvalues_im),
            ("c",              r.c),
        )
            parts = _arr_to_f64(val)
            if length(parts) == 1
                f[name] = parts[1]
            else
                f["$(name)_hi"] = parts[1]
                f["$(name)_lo"] = parts[2]
            end
        end
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

"""
    load_result(path, T)
    load_result(path)

Load an `InvariantDensityResult` from a JLD2 file at `path`.

`T` specifies the numeric type to reconstruct into (`Float64` or `Double64`).
If omitted, `T` is inferred from the `"type"` key written at save time.
"""
function load_result(path::AbstractString, ::Type{T}) where {T<:AbstractFloat}
    jldopen(path, "r") do f
        function read_scalar(name)
            if haskey(f, "$(name)_hi")
                _scalar_from_f64((f["$(name)_hi"]::Float64, f["$(name)_lo"]::Float64), T)
            else
                _scalar_from_f64((f[name]::Float64,), T)
            end
        end

        function read_array(name)
            if haskey(f, "$(name)_hi")
                _arr_from_f64(f["$(name)_hi"], f["$(name)_lo"], T)
            else
                _arr_from_f64(f[name], T)
            end
        end

        InvariantDensityResult{T}(
            read_scalar("alpha"),
            read_scalar("epsilon"),
            f["degree"]           :: Int,
            f["num_break_points"] :: Int,
            f["num_quad_points"]  :: Int,
            f["decouple"] :: Bool,
            read_array("break_points"),
            read_array("M"),
            read_array("G"),
            read_array("eigenvalues_re"),
            read_array("eigenvalues_im"),
            read_array("c"),
        )
    end
end

function load_result(path::AbstractString)
    type_str = jldopen(f -> f["type"] :: String, path, "r")
    T = type_str == "Float64"  ? Float64  :
        type_str == "Double64" ? Double64 :
        error("Unknown numeric type \"$type_str\" in $path. Call load_result(path, T) explicitly.")
    load_result(path, T)
end


# ─────────────────────────────────────────────────────────────────────────────
# Rehydration
# ─────────────────────────────────────────────────────────────────────────────

"""
    RehydratedResult{T,B,P}

Full reconstruction of a saved run: the original `InvariantDensityResult` plus
the live BSplineKit objects rebuilt from it. The basis and spline types are
type parameters (inferred on construction) so that `rh.pdf(x)` dispatches
statically.

Fields
------
- `result`      : the original `InvariantDensityResult{T}`
- `basis`       : `BSplineBasis` reconstructed from `break_points` and `degree`
- `pdf`         : `Spline` for the invariant density (coefficients `c`)
- `eigenvalues` : `Vector{Complex{T}}` of all eigenvalues, sorted |λ| descending
- `decouple`    : if `true`, no B-spline support straddles L or 1-L
"""
struct RehydratedResult{T<:AbstractFloat, B, P}
    result      :: InvariantDensityResult{T}
    basis       :: B
    pdf         :: P
    eigenvalues :: Vector{Complex{T}}
    decouple    :: Bool
end

"""
    rehydrate(result)

Rebuild BSplineKit objects from a stored `InvariantDensityResult`.
Returns a `RehydratedResult` containing the basis, the invariant-density
spline, and the full eigenvalue vector.
"""
function rehydrate(r::InvariantDensityResult{T}) where {T}
    if r.decouple
        L  = Utils.boundary_layer_scale(r.alpha, r.epsilon)
        kv = Bases.build_stacked_knot_vector(copy(r.break_points), T[L, one(T) - L]; p=r.degree)
        B  = BSplineKit.BSplineBasis(BSplineOrder(r.degree + 1), kv; augment=Val(false))
    else
        B = BSplineKit.BSplineBasis(BSplineOrder(r.degree + 1), copy(r.break_points))
    end
    pdf = BSplineKit.Spline(B, copy(r.c))
    λ   = complex.(r.eigenvalues_re, r.eigenvalues_im)
    RehydratedResult(r, B, pdf, λ, r.decouple)
end


# ─────────────────────────────────────────────────────────────────────────────
# Integration of observables against a stored invariant density
# ─────────────────────────────────────────────────────────────────────────────

"""
    density_quadrature(rh::RehydratedResult{T}; n_quad = 32, atol = eps(T),
                       refine = Bases.EndpointRefinement(rh.result.alpha))
    density_quadrature(r::InvariantDensityResult; kwargs...)

Build a `DensityQuadrature` for a stored invariant density, so that

    integrate_against_density(f, dq) ≈ ∫₀¹ f(x) ρ_ε(x) dx.

The mesh is `build_integration_mesh` applied to the *knot* vector of the
rehydrated basis (not `result.break_points`) — that is what carries the stacked
knots when the run used `decouple = true`; `build_star_partition` deduplicates
the repeats.

`refine` defaults to `EndpointRefinement(α)`, matching the map's own `x^α`
endpoint behaviour.  For an observable with a stronger endpoint singularity,
pass a deeper grid, e.g. `Bases.EndpointRefinement(α; τ_end = 1e-18)`.
"""
function Bases.density_quadrature(
    rh     :: RehydratedResult{T};
    n_quad :: Int = 32,
    atol   :: T   = eps(T),
    refine :: Union{Nothing,Bases.EndpointRefinement{T}} =
                 Bases.EndpointRefinement(rh.result.alpha),
) ::Bases.DensityQuadrature{T} where {T<:AbstractFloat}

    knots = collect(T, BSplineKit.knots(rh.basis))
    return Bases.density_quadrature(rh.pdf, knots, rh.result.epsilon;
                                    n_quad=n_quad, atol=atol, refine=refine)
end

Bases.density_quadrature(r::InvariantDensityResult; kwargs...) =
    Bases.density_quadrature(rehydrate(r); kwargs...)


"""
    integrate_against_density(f, rh::RehydratedResult{T}; n_quad = 32, ...) ::T
    integrate_against_density(f, r::InvariantDensityResult{T}; n_quad = 32, ...) ::T

Evaluate `∫₀¹ f(x) ρ_ε(x) dx` for a stored invariant density.  Keyword
arguments are forwarded to `density_quadrature`.

Convenience one-shot form: when integrating several observables against the
same density, build the `DensityQuadrature` once and reuse it instead.
"""
Bases.integrate_against_density(f, rh::RehydratedResult; kwargs...) =
    Bases.integrate_against_density(f, Bases.density_quadrature(rh; kwargs...))

Bases.integrate_against_density(f, r::InvariantDensityResult; kwargs...) =
    Bases.integrate_against_density(f, rehydrate(r); kwargs...)


# ─────────────────────────────────────────────────────────────────────────────
# Autocorrelation curves from a stored run
# ─────────────────────────────────────────────────────────────────────────────

"""
    correlation_curve(f, rh::RehydratedResult{T}, max_lag; n_quad = 32, atol = eps(T),
                      refine = Bases.EndpointRefinement(α))              -> CorrelationCurve{T}
    correlation_curve(f, r::InvariantDensityResult, max_lag; kwargs...)  -> CorrelationCurve{T}

Deterministic stationary autocorrelation of `f` for a stored run,

    C_ε(n) = ∫ f₀(x) · P_ε^n(f₀ ρ_ε)(x) dx,      f₀ = f - ∫ f ρ_ε,

evaluated from the run's own `M`, `G`, and invariant-density coefficients `c`, so
that the curve and the stored spectrum come from the same discretization.

A stored result already carries everything the calculation needs — the operator,
the invariant density, the basis, and ε — so it is the natural "operator +
invariant density" argument.  `refine` defaults to `EndpointRefinement(α)`,
matching `density_quadrature`.  See `Bases.correlation_curve` for the full
argument list and `Bases.CorrelationCurve` for the returned diagnostics.
"""
function Bases.correlation_curve(
    f,
    rh      :: RehydratedResult{T},
    max_lag :: Int;
    n_quad  :: Int = 32,
    atol    :: T   = eps(T),
    refine  :: Union{Nothing,Bases.EndpointRefinement{T}} =
                   Bases.EndpointRefinement(rh.result.alpha),
) ::Bases.CorrelationCurve{T} where {T<:AbstractFloat}

    r    = rh.result
    spls = Bases.build_single_splines(rh.basis)
    return Bases.correlation_curve(f, spls, r.M, r.G, r.c, r.epsilon, max_lag;
                                   n_quad=n_quad, atol=atol, refine=refine)
end

Bases.correlation_curve(f, r::InvariantDensityResult, max_lag::Int; kwargs...) =
    Bases.correlation_curve(f, rehydrate(r), max_lag; kwargs...)
