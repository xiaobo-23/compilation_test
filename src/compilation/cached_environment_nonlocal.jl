# 7/21/2026
# Cached left/right environments for layers of two-site gates at ANY distance (and single-site
# gates) — an adjacent bond is simply the special case of an empty interior. cached_environment.jl
# provides the layer-level helpers dagger_layer / build_psi_right.

using ITensors
using ITensorMPS


# ------- Conventions ------------------------------------------------------------------------------
#
# Layer requirement — gate site-intervals sorted and pairwise disjoint (no interleaving):
#
#     last(idx_pairs[k]) < first(idx_pairs[k+1])
#
# Caches (mirroring cached_environment.jl):
#
#   - ups[k] = ⟨ψ_right|…|ψ_left⟩ transfer contracted over all sites left of gate k's interval,
#     with gates 1..k-1 applied on the fly.
#   - dns[k] = the mirror from the right, with gates k+1..M applied.
#
# Interior sites i+1..j-1 of a two-site gate on (i, j) carry no gate of this layer (disjoint
# intervals), so they enter as bare transfer matrices.
#
# Readout: build_env_nl returns E_T with open legs (s_i, s_i')[, (s_j, s_j')]; the objective is
# real((E_T * gate)[1]), exactly the form consumed by update_rotation_from_env in main.jl.
#
# Cost: all intermediates are O(χ²·d²) — blocks carry at most one open (s, s') pair plus two link
# legs; the gate is contracted only against fully-formed blocks, never stretched across its
# interval as a bare 4-leg tensor.


"""
    check_layer_intervals_nl(idx_pairs)

Assert the layer is representable by this cache: every entry sorted, and
consecutive gate intervals strictly disjoint
(`last(idx_pairs[k]) < first(idx_pairs[k+1])`). Errors otherwise.
"""
function check_layer_intervals_nl(idx_pairs)
    for p in idx_pairs
        issorted(p) || error("gate sites $p are not sorted")
    end
    for k in 1:length(idx_pairs) - 1
        last(idx_pairs[k]) < first(idx_pairs[k + 1]) ||
            error("gate intervals $(idx_pairs[k]) and $(idx_pairs[k + 1]) overlap or interleave — not supported")
    end
end


"""
    init_ups_left_nl!(ups, ψ_left, ψ_right, idx_pairs)

Initialize `ups[1]` with the bare transfer matrices of all sites strictly left
of gate 1's interval (`ITensor(1)` when gate 1 starts at site 1).
"""
function init_ups_left_nl!(ups, ψ_left, ψ_right, idx_pairs)
    T = ITensor(1)
    for m in 1:first(idx_pairs[1]) - 1
        T = T * ψ_left[m] * dag(ψ_right[m])
    end
    ups[1] = T
end


"""
    init_dns_right_nl!(dns, ψ_left, ψ_right, idx_pairs, N)

Initialize `dns[M]` with the bare transfer matrices of all sites strictly right
of the last gate's interval.
"""
function init_dns_right_nl!(dns, ψ_left, ψ_right, idx_pairs, N)
    M = length(idx_pairs)
    T = ITensor(1)
    for m in N:-1:last(idx_pairs[M]) + 1
        T = ψ_left[m] * dag(ψ_right[m]) * T
    end
    dns[M] = T
end


"""
    extend_ups_nl!(ups, ψ_left, ψ_right, gates, idx_pairs, k)

Build `ups[k+1]` from `ups[k]`: absorb gate k (applied on the fly) over its
interval — interior sites enter as bare transfers — then any gap sites before
gate k+1's interval. Call only for `k < M`.
"""
function extend_ups_nl!(ups, ψ_left, ψ_right, gates, idx_pairs, k)
    p = idx_pairs[k]
    if length(p) == 1
        i = p[1]
        T = ups[k] * noprime(gates[k] * ψ_left[i]) * dag(ψ_right[i])
    else
        i, j = p[1], p[2]
        # left block with open (s, s') legs at i and j, then closed by the gate
        L = ups[k] * ψ_left[i] * prime(dag(ψ_right[i]); tags = "Site")
        for m in i + 1:j - 1
            L = L * ψ_left[m] * dag(ψ_right[m])
        end
        L = L * ψ_left[j] * prime(dag(ψ_right[j]); tags = "Site")
        T = L * gates[k]
    end
    for m in last(p) + 1 : first(idx_pairs[k + 1]) - 1
        T = T * ψ_left[m] * dag(ψ_right[m])
    end
    ups[k + 1] = T
end


"""
    contract_dns_from_right_nl!(dns, ψ_left, ψ_right, gates, idx_pairs, k)

Build `dns[k-1]` from `dns[k]`: absorb gate k over its interval, then any gap
sites down to gate k-1's interval. Call only for `k > 1`.
"""
function contract_dns_from_right_nl!(dns, ψ_left, ψ_right, gates, idx_pairs, k)
    p = idx_pairs[k]
    if length(p) == 1
        i = p[1]
        T = noprime(gates[k] * ψ_left[i]) * dag(ψ_right[i]) * dns[k]
    else
        i, j = p[1], p[2]
        R = ψ_left[j] * prime(dag(ψ_right[j]); tags = "Site") * dns[k]
        for m in j - 1:-1:i + 1
            R = ψ_left[m] * dag(ψ_right[m]) * R
        end
        R = ψ_left[i] * prime(dag(ψ_right[i]); tags = "Site") * R
        T = R * gates[k]
    end
    for m in first(p) - 1 : -1 : last(idx_pairs[k - 1]) + 1
        T = ψ_left[m] * dag(ψ_right[m]) * T
    end
    dns[k - 1] = T
end


"""
    precompute_dns_nl!(dns, ψ_left, ψ_right, gates, idx_pairs, N)

Fill `dns[M..1]` once per layer call: `init_dns_right_nl!`, then a
right-to-left walk with `contract_dns_from_right_nl!`.
"""
function precompute_dns_nl!(dns, ψ_left, ψ_right, gates, idx_pairs, N)
    M = length(idx_pairs)
    init_dns_right_nl!(dns, ψ_left, ψ_right, idx_pairs, N)
    for k in M:-1:2
        contract_dns_from_right_nl!(dns, ψ_left, ψ_right, gates, idx_pairs, k)
    end
end


"""
    build_env_nl(ups, dns, ψ_left, ψ_right, k, idx_pairs) -> ITensor

Environment of gate k — everything except the gate itself — with open legs
`(s_i, s_i')` (single-site) or `(s_i, s_i', s_j, s_j')` (two-site; interior
sites bridged by bare transfers). `real((E_T * gate)[1])` is the global
overlap objective.
"""
function build_env_nl(ups, dns, ψ_left, ψ_right, k, idx_pairs)
    p = idx_pairs[k]
    if length(p) == 1
        i = p[1]
        return ups[k] * ψ_left[i] * prime(dag(ψ_right[i]); tags = "Site") * dns[k]
    else
        i, j = p[1], p[2]
        L = ups[k] * ψ_left[i] * prime(dag(ψ_right[i]); tags = "Site")
        for m in i + 1:j - 1
            L = L * ψ_left[m] * dag(ψ_right[m])
        end
        return L * ψ_left[j] * prime(dag(ψ_right[j]); tags = "Site") * dns[k]
    end
end
