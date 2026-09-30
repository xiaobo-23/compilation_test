# 5/21/2026
# Layer-level environment helpers for the sweep optimization: `dagger_layer` folds a layer into
# the bra side, and `build_psi_right` constructs the right-environment collection consumed layer
# by layer in main.jl. The per-gate environment caches live in cached_environment_nonlocal.jl.

using ITensors
using ITensorMPS



"""
    dagger_layer(gates) -> Vector{ITensor}

Hermitian-conjugate a layer of gates: each `G ↦ G†` via `swapprime(dag(g), 0 => 1)`.
Used to fold a layer into the bra side — `⟨ψ_T| G` written as the ket `G† |ψ_T⟩` —
as in `build_psi_right`. Non-mutating: the input `gates` (and the circuit they
belong to) are left untouched.
"""
dagger_layer(gates) = [swapprime(dag(g), 0 => 1) for g in gates]



"""
    build_psi_right(circuit_gates, ψ_T; cutoff) -> Vector{MPS}

Right-environment collection for the layer-by-layer sweep: `coll[idx]` is `ψ_T`
with every layer *above* `idx` folded in from the bra side — exactly the `ψ_right`
used when optimizing layer `idx` (layer `idx` itself excluded). Built incrementally
in `O(nlayers)` MPS applies, normalizing after each; every entry is a fresh,
independent `MPS` (`coll[nlayers]` a `deepcopy` of `ψ_T`). Rebuilt once per growth
stage in main.jl; within each sweep the backward pass then updates `ψ_right` on the fly.
"""
function build_psi_right(circuit_gates, ψ_T; cutoff)
    nlayers = length(circuit_gates)
    coll = Vector{MPS}(undef, nlayers)
    coll[nlayers] = deepcopy(ψ_T)         # independent copy, not an alias
    for idx in nlayers - 1 : -1 : 1
        coll[idx] = normalize!(apply(dagger_layer(circuit_gates[idx + 1]),
                                     coll[idx + 1]; cutoff))
    end
    return coll
end
