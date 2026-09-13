using Roots: find_zero, Halley, Bisection, Newton, Brent, FalsePosition


#-------------------------------------------------------------------------------
# Generic Inverse Maker
#-------------------------------------------------------------------------------

"""
    make_inverse(f, a, b; monotone=:auto, method=:brent, atol=1e-12, rtol=1e-12, maxevals=10_000)

Return a function `invf(y)` that finds `x ∈ [a,b]` with `f(x) = y`, using Roots.jl find_zero().

Assumptions:
- `f` is continuous and (strictly) monotone on `[a,b]`. If not, split into monotone branches and
  call `make_inverse` on each subinterval.
- `y` must lie in `f([a,b])`.

Arguments:
- `monotone`: `:auto` (default), `:inc`, or `:dec`. `:auto` infers from `f(a)` and `f(b)`.
  Any other symbol is an error rather than a silent fall-back to `:auto`.
- `method`: `:brent` (robust & fast), `:bisection` (guaranteed), or `:falseposition`.
- `atol`, `rtol`, `maxevals`:  These arguments are input into find_zero. See Roots.jl `find_zero`
    documentation.
"""
function make_inverse(f, a::Real, b::Real;
                      monotone::Symbol = :auto,
                      method::Symbol = :brent,
                      atol::Real = 1e-12,
                      rtol::Real = 1e-12,
                      maxevals::Int = 10_000)

    a < b || throw(ArgumentError("Require a < b; got a=$a, b=$b"))
    monotone === :auto || monotone === :inc || monotone === :dec ||
        throw(ArgumentError("monotone must be :auto, :inc or :dec, got :$monotone"))

    fa = f(a); fb = f(b)
    inc = monotone === :inc ? true :
          monotone === :dec ? false :
          fa <= fb                   # :auto, the only symbol left

    ymin, ymax = inc ? (fa, fb) : (fb, fa)

    mobj = method === :bisection     ? Bisection() :
           method === :brent         ? Brent() :
           method === :falseposition ? FalsePosition() :
           throw(ArgumentError("Unknown method: $method"))

    function invf(y::Real; a_bracket::Real=a, b_bracket::Real=b)

        if y < ymin || y > ymax
            throw(DomainError(y, "Target y not in range f([a,b]) = [$ymin, $ymax]."))
        end

        y == fa && return a
        y == fb && return b

        g(x) = f(x) - y
        x = find_zero(g, (a_bracket, b_bracket), mobj; atol=atol, rtol=rtol, maxevals=maxevals)
        return x
    end

    return invf
end


#-------------------------------------------------------------------------------
# Inverse Table
#-------------------------------------------------------------------------------

"""
    InverseTable{T}

Warm-start table of an inverse x = f^{-1}(y) for a strictly monotone map.
- `y`         : strictly increasing knots in the image space of f
- `x`         : inverse at the knots
- `xp`        : 1 / f′(x) at each knot
- `prec_bits` : BigFloat precision used during table build if applicable
"""
struct InverseTable{T<:AbstractFloat}
    y         :: Vector{T}
    x         :: Vector{T}
    xp        :: Vector{T}
    prec_bits :: Union{Int,Nothing}
end


"""
    retarget_table(table::InverseTable, ::Type{T}) -> InverseTable{T}

Convert an InverseTable to numeric type `T` (Float64, Double64, BigFloat, …).
"""
function retarget_table(table::InverseTable{<:AbstractFloat}, ::Type{T}) where {T<:AbstractFloat}
    return InverseTable{T}(T.(table.y), T.(table.x), T.(table.xp), table.prec_bits)
end


# We write our own bisection method to get access to lower/upper bound at prior steps. This is
# useful for writing a bisection then polish strategy.
"""
    bisection(f, a::T, b::T; kwargs...) where {T<:AbstractFloat}
    bisection(f, a::T, b::T, fa::T, fb::T; rtol::T = sqrt(eps(T)), atol::T = zero(T), maxiters::Int = 10_000) where {T<:AbstractFloat}

Finds a root of `f` on `[a,b]` by bisection, assuming `f(a)` and `f(b)` have
opposite signs (or one endpoint is an exact root).

The 5-argument method accepts pre-computed bounds `fa = f(a)` and `fb = f(b)`.

Returns a `NamedTuple{(:x, :a, :b, :fx, :iterations, :converged)}`:
- `x` : the current midpoint
- `a,b` : the final bracketing interval with a sign change (safe guard),
- `fx` : `f(x)`,
- `iterations` : number of bisection steps taken,
- `converged` : whether the stopping criterion was met.

Stopping criterion: `(b - a) ≤ max(atol, rtol * max(abs(a), abs(b)))` or `f(x) == 0`.

Notes:
- Bracket is maintained at every step; if an exact zero is found, the interval collapses to `[x,x]`.
"""
function bisection end

function bisection(f, a::T, b::T, fa::T, fb::T; rtol::T = sqrt(eps(T)), atol::T = zero(T), maxiters::Int = 10_000) where {T<:AbstractFloat}
    isfinite(a) && isfinite(b) || throw(ArgumentError("Endpoints must be finite."))
    a < b || throw(ArgumentError("Require a < b."))

    left, right = a, b
    fl, fr = fa, fb

    if fl == zero(T)
        return (x=left, a=left, b=left, fx=zero(T), iterations=0, converged=true)
    elseif fr == zero(T)
        return (x=right, a=right, b=right, fx=zero(T), iterations=0, converged=true)
    end
    xor(signbit(fl), signbit(fr)) ||
        throw(ArgumentError("f(a) and f(b) must have opposite signs (bracketing required)."))

    mid, fm = left, fl
    for k in 1:maxiters
        mid = left + (right - left)/T(2)
        fm  = f(mid)

        if fm == zero(T)
            return (x=mid, a=mid, b=mid, fx=zero(T), iterations=k, converged=true)
        end
        if (right - left) ≤ max(atol, rtol * max(abs(left), abs(right)))
            return (x=mid, a=left, b=right, fx=fm, iterations=k, converged=true)
        end

        # The root must lie in either [left, mid] or [mid, right]
        if xor(signbit(fl), signbit(fm)) # Does the root lie in [left, mid]?
            right = mid
            fr = fm
        else                             # Otherwise, it must lie in [mid, right]
            left = mid
            fl = fm
        end
    end

    mid = left + (right - left)/T(2)
    fm  = f(mid)
    return (x=mid, a=left, b=right, fx=fm, iterations=maxiters, converged=false)
end


function bisection(f, a::T, b::T; kwargs...) where {T<:AbstractFloat}
    return bisection(f, a, b, f(a), f(b); kwargs...)
end


#-------------------------------------------------------------------------------
# PM-Specific Map Generator
#-------------------------------------------------------------------------------

"""
    make_left_branch_maps(α::Real, ::Type{T}) where {T<:AbstractFloat}

Return `(f, df, ddf)` — single-argument closures for the left branch of the symmetric PM map
`f(x) = x + 2^α x^(1+α)` and its first two derivatives, with all constants and computation
in type `T`.

Type purity: all captured constants (`αT`, `e2α`, `oneT`) are created in `T`; each closure
argument is typed `x::T`. When called with `T=BigFloat` inside a `setprecision(BigFloat, p)` block,
the captured BigFloat constants are created at precision `p`.
"""
function make_left_branch_maps(α::Real, ::Type{T}) where {T<:AbstractFloat}
    αT   = T(α)
    oneT = one(T)
    e2α  = exp2(αT)
    f(x::T)   = x + e2α * x^(oneT + αT)
    df(x::T)  = oneT + (oneT + αT) * e2α * x^αT
    ddf(x::T) = αT * (oneT + αT) * e2α * x^(αT - oneT)
    return (f=f, df=df, ddf=ddf)
end


#-------------------------------------------------------------------------------
# Build Inverse Table (General)
#-------------------------------------------------------------------------------

"""
    build_inverse_table(a, b, map_gen;
        N=5000, prec_bits=256,
        x_bisect_atol=1e-6,
        bisect_maxiters=10_000,
        polish_retries=20,
        roots_kwargs...) -> InverseTable{BigFloat}

Construct a BigFloat inverse table for any strictly increasing C^2 map on [a, b] using a
bracket–then–polish strategy.

ASSUMES THE MAP IS STRICTLY INCREASING ON [a, b].

Inputs/Options:
- `a, b`           : domain endpoints
- `map_gen`        : `(::Type{T}) -> (f=..., df=..., ddf=...)` returning single-arg typed closures
                     for the map and its first two derivatives. Called as `map_gen(BigFloat)` inside
                     the `setprecision` block so BigFloat constants in the closures are created at
                     precision `prec_bits`.
- `N`              : number of table knots (uniform in image space)
- `prec_bits`      : BigFloat precision used during the build
- `x_bisect_atol`  : absolute x-tolerance for the bisection phase
- `bisect_maxiters`: cap on bisection iterations
- `polish_retries` : max number of (single) bisection-step reseeds if the
                     polished root exits [L₀, R₀]
- `roots_kwargs...`: extra keywords forwarded to `find_zero` (e.g.,
                     `xatol`, `rtol`, `abstol`, `verbose`, `maxiters`, etc.).

Returns:
- `InverseTable{BigFloat}(y, x, xp, prec_bits)` where
   * `y`  : monotonically increasing image-grid values
   * `x`  : f⁻¹(y) at each knot
   * `xp` : derivative dx/dy = 1 / f′(x)
"""
function build_inverse_table(a::Real, b::Real, map_gen;
        N::Integer=5000,
        prec_bits::Integer=256,
        x_bisect_atol::Real = 1e-6,
        bisect_maxiters::Integer = 10_000,
        polish_retries::Integer = 20,
        roots_kwargs...)

    N ≥ 4 || throw(ArgumentError("Need at least 4 knots for a useful table"))

    setprecision(BigFloat, prec_bits) do
        # map_gen called inside setprecision so BigFloat constants are at prec_bits precision
        maps = map_gen(BigFloat)
        Tf, dTf, ddTf = maps.f, maps.df, maps.ddf

        a_bf = BigFloat(a)
        b_bf = BigFloat(b)

        ya, yb = Tf(a_bf), Tf(b_bf)
        ymin, ymax = minmax(ya, yb)
        y  = [ymin + BigFloat(i-1)*(ymax - ymin)/BigFloat(N-1) for i in 1:N]
        x  = similar(y)
        xp = similar(y)

        x_bis_atolBF = BigFloat(x_bisect_atol)
        zeroBF = zero(BigFloat)

        prev_x = a_bf
        for k in 1:N
            yk = y[k]
            g(x::BigFloat) = Tf(x) - yk

            # Monotone bracket: [a, b] for k==1; [prev_x, b] for k>1
            L_init  = (k == 1) ? a_bf : prev_x
            fL_init = g(L_init)
            fb_val  = yb - yk

            @assert fL_init ≤ zeroBF "Monotone bracket left endpoint must satisfy f(L)-y ≤ 0"
            @assert fb_val  ≥ zeroBF "Right endpoint must satisfy f(b)-y ≥ 0"

            bis = bisection(g, L_init, b_bf, fL_init, fb_val;
                            rtol=BigFloat(0), atol=x_bis_atolBF, maxiters=bisect_maxiters)

            L0, R0 = bis.a, bis.b
            fL0, fR0 = g(L0), g(R0)
            x0 = (L0 + R0)/big(2)

            # Full polish with Roots.jl; enforce that final iterate stays within the original [L0, R0]
            accepted = false
            xhat = x0
            L, R = L0, R0
            fL, fR = fL0, fR0

            for _ in 1:polish_retries
                # try/catch as an expression: NaN signals both polishers failed
                xhat_try = try
                    find_zero((g, dTf, ddTf), x0, Halley(); roots_kwargs...)
                catch
                    try
                        find_zero((g, dTf), x0, Newton(); roots_kwargs...)
                    catch
                        BigFloat(NaN)
                    end
                end

                if isfinite(xhat_try) && L ≤ xhat_try ≤ R
                    xhat = xhat_try
                    accepted = true
                    break
                end

                # Not acceptable: do ONE bisection step on [L, R], reseed x0, retry
                mid = L + (R - L)/big(2)
                fm  = g(mid)
                if fm == zeroBF
                    xhat = mid
                    accepted = true
                    break
                end

                if xor(signbit(fL), signbit(fm))
                    R = mid; fR = fm
                else
                    L = mid; fL = fm
                end
                x0 = (L + R)/big(2)
            end

            accepted || error("Polish failed to produce an in-bracket solution at knot $k.")

            x[k]  = xhat
            xp[k] = inv(dTf(xhat))
            prev_x = xhat
        end

        return InverseTable{BigFloat}(y, x, xp, prec_bits)
    end
end


#-------------------------------------------------------------------------------
# Generate Inverse Function
#-------------------------------------------------------------------------------

"""
    warmstart(y::T, tbl::InverseTable{T}) -> (kL::Int, kR::Int, x0::T)

Generate a warm-start guess for the inverse x = f⁻¹(y) using a precomputed
`InverseTable`.

Algorithm:
1. Find the adjacent knot indices `(kL, kR)` such that `tbl.y[kL] ≤ y ≤ tbl.y[kR]`.
2. Choose the nearer knot in y-space and apply a first-order inverse linearization:
       x0 ≈ x_k + (y - y_k) / f′(x_k)
   where `(y_k, x_k, f′(x_k))` are taken from the selected knot.

Returns
-------
- `kL, kR` : indices of the bracketing knots in the table.
- `x0`     : type-`T` initial guess for the inverse at `y`.
"""
@inline function warmstart(y::T, tbl::InverseTable{T}) where {T<:AbstractFloat}
    ys, xs, xps = tbl.y, tbl.x, tbl.xp
    N = length(ys)
    N ≥ 2 || throw(ArgumentError("Table must have at least two knots"))
    ys[1] <= y <= ys[end] || throw(DomainError(y, "y is out of the table's range."))

    # First index i with ys[i] ≥ y  (1 .. N+1)
    i = searchsortedfirst(ys, y)

    # Choose a valid bracketing pair (kL, kR) in 1..N-1
    kL = clamp(i-1, 1, N-1)
    kR = kL + 1

    # Inverse linearization at the nearer knot
    dL = abs(y - ys[kL])
    dR = abs(ys[kR] - y)
    x0 = if dL <= dR
        xs[kL] + xps[kL] * (y - ys[kL])
    else
        xs[kR] + xps[kR] * (y - ys[kR])
    end

    return kL, kR, x0
end


"""
    inverse_eval(y::T, tbl::InverseTable{T}, f, df, ddf;
                 halley_kw::NamedTuple=NamedTuple(),
                 newton_kw::NamedTuple=NamedTuple(),
                 bisection_kw::NamedTuple=NamedTuple()) where {T<:AbstractFloat}

Evaluate the inverse x = f⁻¹(y) using warm start from `tbl` and polishing with Roots.jl.

ASSUMES THAT f IS STRICTLY INCREASING ON THE TABLE'S DOMAIN.

Inputs:
- `y`               : point at which to evaluate the inverse
- `tbl`             : `InverseTable{T}` to be used for warm start and bracketing
- `f`               : forward map `x::T -> f(x)::T`
- `df`              : derivative `x::T -> f′(x)::T`
- `ddf`             : second derivative `x::T -> f″(x)::T`
- `halley_kw`       : kwargs for Halley polishing
- `newton_kw`       : kwargs for Newton polishing
- `bisection_kw`    : kwargs for bracketed bisection

Returns:
- `x̂` : approximate inverse `f⁻¹(y)` in type `T`
"""
function inverse_eval(y::T, tbl::InverseTable{T}, f, df, ddf;
                      halley_kw::NamedTuple=NamedTuple(),
                      newton_kw::NamedTuple=NamedTuple(),
                      bisection_kw::NamedTuple=NamedTuple()) where {T<:AbstractFloat}

    ys, xs = tbl.y, tbl.x
    N      = length(ys)

    N ≥ 2 || throw(ArgumentError("Table must have at least two knots"))
    isfinite(y) || throw(ArgumentError("y must be finite"))
    ys[1] ≤ y ≤ ys[end] || throw(DomainError(y, "y is out of the table's range."))

    # Warm start & bracket from table
    kL, kR, x0 = warmstart(y, tbl)
    xL, xR     = xs[kL], xs[kR]
    @assert xL ≤ xR "warmstart produced an invalid bracket"   # internal invariant

    y == ys[kL] && return xs[kL]
    y == ys[kR] && return xs[kR]

    g(x::T) = T(f(x) - y)

    # Halley → Newton → bracketed bisection, matching build_inverse_table's polish strategy
    x̂ = try
        find_zero((g, df, ddf), x0, Halley(); halley_kw...)
    catch
        try
            find_zero((g, df), x0, Newton(); newton_kw...)
        catch
            find_zero(g, (xL, xR), Bisection(); bisection_kw...)
        end
    end

    # Safety: if out of bracket or non-finite, finish with bracketed solve
    if !(isfinite(x̂) && xL ≤ x̂ ≤ xR)
        x̂ = find_zero(g, (xL, xR), Bisection(); bisection_kw...)
    end

    return x̂
end


"""
    TabulatedInverse{T,F1,F2,F3,IT}

Callable container for evaluating the inverse of a strictly increasing C^2 map,
backed by a precomputed warm-start table.

Fields:
- `f::F1`                   : forward map `x::T -> f(x)::T`
- `df::F2`                  : derivative `x::T -> f′(x)::T`
- `ddf::F3`                 : second derivative `x::T -> f″(x)::T`
- `tbl::IT`                 : `InverseTable{T}` with knots `(y, x, xp)`
- `newton_kw::NamedTuple`   : kwargs for Newton polishing
- `halley_kw::NamedTuple`   : kwargs for Halley polishing
- `bisection_kw::NamedTuple`: kwargs for bracketed bisection
"""
struct TabulatedInverse{T<:AbstractFloat,F1,F2,F3,IT<:InverseTable{T}}
    f            :: F1
    df           :: F2
    ddf          :: F3
    tbl          :: IT
    newton_kw    :: NamedTuple
    halley_kw    :: NamedTuple
    bisection_kw :: NamedTuple
end


@inline function (ri::TabulatedInverse{T})(y::T) where {T<:AbstractFloat}
    inverse_eval(y, ri.tbl, ri.f, ri.df, ri.ddf;
        newton_kw    = ri.newton_kw,
        halley_kw    = ri.halley_kw,
        bisection_kw = ri.bisection_kw)
end


#------------------------------------------------------
# Public constructors
#------------------------------------------------------

"""
    construct_tabulated_inverse(a::T, b::T, f, df, ddf;
        n_knots::Integer = 5_000,
        newton_kw::NamedTuple = NamedTuple(),
        halley_kw::NamedTuple = NamedTuple(),
        bisection_kw::NamedTuple = NamedTuple()
    ) where {T<:AbstractFloat}

Construct a `TabulatedInverse{T}` for any strictly increasing C^2 map on `[a, b]`.

Arguments:
- `a, b`   : domain endpoints; their type `T` sets the working precision.
- `f`      : the map — must work for any `AbstractFloat` argument (avoid hardcoded `Float64` literals
             inside it so the high-precision BigFloat table build is accurate).
- `df`     : first derivative of `f`.
- `ddf`    : second derivative of `f`.

The table is built in BigFloat at 256-bit precision and retargeted to `T`.
Returns a `TabulatedInverse{T}` callable as `inv_f(y::T) -> T`.
"""
function construct_tabulated_inverse(a::T, b::T, f, df, ddf;
    n_knots::Integer = 5_000,
    newton_kw::NamedTuple = NamedTuple(),
    halley_kw::NamedTuple = NamedTuple(),
    bisection_kw::NamedTuple = NamedTuple()
) where {T<:AbstractFloat}

    # Wrap user's functions into output-typed closures for any precision S.
    # Argument type annotations require a where-clause type param, not a lambda arg,
    # so we rely on S(...) output conversion for type purity.
    gen = S -> (f   = x -> S(f(x)),
                df  = x -> S(df(x)),
                ddf = x -> S(ddf(x)))

    maps = gen(T)
    tbl  = retarget_table(build_inverse_table(a, b, gen; N=n_knots), T)

    return TabulatedInverse{T,typeof(maps.f),typeof(maps.df),typeof(maps.ddf),typeof(tbl)}(
        maps.f, maps.df, maps.ddf, tbl, newton_kw, halley_kw, bisection_kw)
end


"""
    construct_left_branch_inverse(α::T;
        table::Union{Nothing,InverseTable{<:AbstractFloat}}=nothing,
        n_knots::Integer = 5_000,
        newton_kw::NamedTuple = NamedTuple(),
        halley_kw::NamedTuple = NamedTuple(),
        bisection_kw::NamedTuple = NamedTuple()
    ) where {T<:AbstractFloat}

Returns a `TabulatedInverse{T}` for the left branch of the symmetric PM map
f(x; α) = x + 2^α x^(1+α) on x ∈ [0, 1/2], using a warm-start table to speed up computation. The
returned object can be called as a function to evaluate the inverse at any `y` in the image of f.
"""
function construct_left_branch_inverse(α::T;
    table::Union{Nothing,InverseTable{<:AbstractFloat}}=nothing,
    n_knots::Integer = 5_000,
    newton_kw::NamedTuple = NamedTuple(),
    halley_kw::NamedTuple = NamedTuple(),
    bisection_kw::NamedTuple = NamedTuple()
) where {T<:AbstractFloat}

    # Generator closes over α: given any type S, returns maps typed in S
    map_gen = S -> make_left_branch_maps(α, S)

    # Maps at working precision T
    maps = map_gen(T)

    # Build or retarget the table (built in BigFloat at high precision, then converted to T)
    tblT = if table === nothing
        retarget_table(build_inverse_table(T(0), T(0.5), map_gen; N=n_knots), T) # build_inverse_table returns BigFloat table, retarget to T
    elseif table isa InverseTable{T}
        table
    else
        retarget_table(table, T)
    end

    return TabulatedInverse{T,typeof(maps.f),typeof(maps.df),typeof(maps.ddf),typeof(tblT)}(
        maps.f, maps.df, maps.ddf, tblT, newton_kw, halley_kw, bisection_kw)
end
