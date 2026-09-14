# Copyright (c) 2026 Ben Stilin
#
# Licensed under the MIT License. See LICENSE for details.

"""
_check_nonwrapping(a::T, b::T) where {T<:AbstractFloat}

Validates 0 ≤ a < b ≤ 1 (non-wrapping). Throws ArgumentError otherwise.
"""
function _check_nonwrapping(a::T, b::T) where {T<:AbstractFloat}
    (0 ≤ a ≤ 1 && 0 ≤ b ≤ 1 && a < b) ||
        throw(ArgumentError("Expected non-wrapping [a,b] ⊂ [0,1] with a<b; got a=$a, b=$b"))
    return nothing
end


"""
union_intervals(sorted::Vector{Tuple{T,T}}; atol::T = eps(T)) where {T}

Merge a sorted list of non-wrapping intervals on `[0,1]` by coalescing all
overlapping or adjacent intervals.

Each element of `sorted` must be a pair `(a,b)` with `0 ≤ a < b ≤ 1`, and the
vector must be sorted by `a`. The function returns a new vector of disjoint
intervals covering the same total set.

Merging rule:
- If the next interval `(a,b)` starts at or before the current right endpoint
`cur_b` (within `eps(T)` tolerance), extend: `cur_b = max(cur_b,b)`.
- Otherwise, push the current interval and start a new one.

Notes
-----
- We choose atol=eps(T) for adjacency tolerance as all intervals are assumed to be
  in [0,1].

Returns
-------
`Vector{Tuple{T,T}}` : A minimal list of disjoint, non-wrapping intervals on
`[0,1]` representing the union of `sorted`.


Examples
--------
```julia
union_intervals([(0.1,0.2), (0.15,0.3), (0.5,0.6)])
# => [(0.1,0.3), (0.5,0.6)]


union_intervals([(0.1,0.2), (0.2,0.25)])
# => [(0.1,0.25)]
```
"""
function union_intervals(sorted::Vector{NTuple{2,T}}; atol::T = eps(T)) ::Vector{NTuple{2,T}} where {T<:AbstractFloat}
    isempty(sorted) && return sorted

    out = NTuple{2,T}[]
    cur_a, cur_b = sorted[1]
    for k in 2:length(sorted)
        a, b = sorted[k]

        a >= b && throw(ArgumentError("Intervals assumed to satisfy a < b."))

        if a ≤ cur_b || isapprox(a, cur_b; atol=atol)
            cur_b = max(cur_b, b)
        else
            push!(out, (cur_a, cur_b))
            cur_a, cur_b = a, b
        end
    end
    push!(out, (cur_a, cur_b))
    return out
end


"""
    intersect_intervals!(I, J; out, atol=eps(T))

Compute the intersections of two lists of 1D intervals, writing results into
`out` without allocating.

Arguments
- `I`, `J` — Vectors of `(a,b)` with `a < b`, element type `T<:AbstractFloat`.
  Each must be sorted by start point and contain non-overlapping intervals.
- `out` — Preallocated vector to receive intersections (emptied on entry).
- `atol` — Absolute tolerance; intersections with width ≤ `atol` are discarded.

Returns
- `out`, filled with `(L,R)` where `L = max(a,c)` and `R = min(b,d)`.
  Shared endpoints are treated as empty up to `atol`.
"""
function intersect_intervals!(I::Vector{NTuple{2,T}},
                              J::Vector{NTuple{2,T}};
                              out::Vector{NTuple{2,T}},
                              atol::T = eps(T)) where {T<:AbstractFloat}
    empty!(out)
    i = 1; j = 1
    while i ≤ length(I) && j ≤ length(J)
        a, b = I[i]
        c, d = J[j]
        L = max(a, c)
        R = min(b, d)
        if L < R && !isapprox(L, R; atol=atol)
            push!(out, (L, R))
        end
        if b < d
            i += 1
        else
            j += 1
        end
    end
    return out
end


"""
    mod_interval(interval::Tuple{T,T}) where {T<:AbstractFloat}

Reduce a real interval `(a, b)` to an equivalent, non-degenerate interval on `[0,1]`

The function ensures the result represents a single, continuous sub-arc of the
circle (i.e., it does not wrap across the seam at 0 ≡ 1) and preserves orientation.
Endpoints at 0 or 1 are handled so that the interval lies cleanly within `[0,1]`.
We assume that:

- The interval is non-degenerate: a < b
- The interval does not cross 0 or 1: !(a < 0 < b) and !(a < 1 < b)
- The interval is within (-1,2): a > -1 and b < 2

Assumptions guarantee that a single input interval is mapped to a single output tuple.

Examples
--------

```julia
mod_interval((-0.3, 0.0))   
    # => (0.7, 1.0)

mod_interval((1.0, 1.2))    
    # => (0.0, 0.2)

mod_interval((0.2, 0.5))    
    # => (0.2, 0.5)
```
"""
function mod_interval(interval::Tuple{T,T}) where {T<:AbstractFloat}

    oneT = one(T)
    zeroT = zero(T)
    twoT = T(2)
    a, b = interval

    a<b && !(a<zeroT && b>zeroT)  && !(a<oneT && b>oneT) && a>-oneT && b < twoT ||
        throw(ArgumentError("Interval must be non-degenerate See docstring."))

    if a==oneT
        return (zeroT,mod(b, oneT))
    elseif b==oneT
        return (mod(a, oneT), oneT)
    elseif a==zeroT
        return (zeroT, mod(b, oneT))
    elseif b==zeroT
        return (mod(a, oneT), oneT)
    else
        return minmax(mod(a, oneT), mod(b, oneT))
    end
end



"""
    split_circular!(out, t, ε; atol = eps(T))

Append to `out::Vector{Tuple{T,T}}` the non-wrapping pieces in `[0,1]` that
cover the circular interval `(t-ε, t+ε)` (mod 1), specialized to symmetric windows.

Assumptions (enforced):
- `0 ≤ t ≤ 1`
- `0 ≤ ε < 1/2`

Guarantees:
- Each pushed interval `(a,b)` satisfies `0 ≤ a < b ≤ 1` (strict a<b).
- If `2ε ≤ atol`, nothing is pushed.

Cases:
- No wrap (`t-ε ≥ 0` and `t+ε ≤ 1`): push `(t-ε, t+ε)`.
- Wrap-left (`t-ε < 0`): push `(0, t+ε)` and `(1-(ε-t), 1)`.
- Wrap-right (`t+ε > 1`): push `(0, t+ε-1)` and `(t-ε, 1)`.
"""


function split_circular!(out::Vector{NTuple{2,T}}, t::T, ε::T; atol::T = eps(T)) where {T<:AbstractFloat}
    zero(T) ≤ t ≤ one(T) || throw(ArgumentError("t must be in [0,1], got $t"))
    zero(T) ≤ ε < T(0.5) || throw(ArgumentError("ε must satisfy 0 ≤ ε < 1/2, got $ε"))

    # drop empty/tiny
    if 2ε ≤ atol
        return out
    end

    a = t - ε
    b = t + ε

    if a ≥ zero(T) && b ≤ one(T)
        # no wrap
        push!(out, (a, b))                    # a<b since ε>0
    elseif a < zero(T)
        # wrap across left seam
        push!(out, (zero(T), b))              # 0 < b ≤ 1
        push!(out, (one(T) + a, one(T)))      # 0 ≤ 1+a < 1
    else
        # wrap across right seam (must have b>1 here)
        push!(out, (zero(T), b - one(T)))     # 0 < b-1 < 1
        push!(out, (a, one(T)))               # a < 1
    end
    return out
end


"""
    overlaps_any(a, b, intervals) -> Bool
    overlaps_any(a, b, interval::Tuple)  -> Bool

Return `true` if the interval `[a, b]` overlaps at least one interval in
`intervals` (a collection of `(c, d)` tuples), or a single `(c, d)` tuple.
Two intervals overlap when `max(a, c) < min(b, d)`.
"""
overlaps_any(a, b, intervals) = any(max(a, c) < min(b, d) for (c, d) in intervals)
overlaps_any(a, b, interval::Tuple) = max(a, interval[1]) < min(b, interval[2])