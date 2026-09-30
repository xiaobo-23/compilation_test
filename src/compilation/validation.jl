# 4/27/2026
# Measurement & validation for compiled Kitaev cluster wave functions.
# Everything geometry-specific is bundled by a builder into one "geometry
# context" NamedTuple:
#
#     geom = (; H, plaquettes, plaquette_ops[, loops, loop_ops])
#
#   - H             :: MPO                  the Kitaev Hamiltonian for energy
#   - plaquettes    :: Vector{Vector{Int}} six-site index list per hexagon
#   - plaquette_ops :: NTuple/Vector        Pauli string matched to the vertex order
#                                           (one shared pattern, or one string per plaquette)
#   - loops/loop_ops (optional)             non-contractible loop strings; when present the
#                                           validators also report ⟨L⟩ (topological sector)
#
# main.jl builds the context inline from its generated plaquette/loop tables,
# then applies `validate_circuit` / `validate_reference`. The local
# `cluster_energy_mpo(...; pauli = false)` also supports the legacy spin-1/2
# convention of the older stored targets.

using ITensors
using ITensorMPS

include("../helper/honeycomb_lattice.jl")   # LatticeBond/WedgeBond/Lattice + honeycomb lattice/wedge builders
include("../helper/hamiltonian.jl")         # cluster_energy_mpo (spin-1/2 & Pauli conventions)



# ------- Plaquette measurement --------------------------------------------------------------------
"""
    plaquette_sign(ops) -> Float64

Sign correction for the `iY` convention: `(iY)ⁿ = iⁿ`, and `n` (number of
`iY` factors) is even ⇒ ±1. Equals −1 for the standard two-`iY` strings.
"""
plaquette_sign(ops) = real(im ^ count(==("iY"), ops))


"""
    plaquette_mpo(p_sites, sites, ops) -> MPO

Build the plaquette MPO `∏ₖ ops[k](p_sites[k])` on `sites`. The six operators
act on distinct sites, so the product is order-independent; what matters is the
(`ops[k]` → `p_sites[k]`) pairing, which is fixed by the geometry's generator.
"""
function plaquette_mpo(p_sites, sites, ops)
    os = OpSum()
    add!(os, 1.0, Iterators.flatten(zip(ops, p_sites))...)
    return MPO(os, sites)
end


"""
    measure_plaquettes(ψ, sites, strings, ops_per_string::Vector; imag_tol = 1e-8)

Per-string ⟨Wp⟩ / ⟨L⟩: one operator list per site list — main.jl's generated
plaquette and loop tables both use this path. In the flux-free Kitaev ground
state every ⟨Wp⟩ = +1, so closeness to +1 is a local flux-sector check that
holds even at moderate global fidelity. Warns if any contraction has a
non-negligible imaginary part (the strings are Hermitian, so values should
be real).
"""
function measure_plaquettes(ψ::MPS, sites, plaquettes, ops::Vector{<:AbstractVector};
                            imag_tol::Real = 1e-8)
    length(ops) == length(plaquettes) ||
        error("got $(length(ops)) operator strings for $(length(plaquettes)) site lists")
    vals = Vector{Float64}(undef, length(plaquettes))
    for (k, (p, o)) in enumerate(zip(plaquettes, ops))
        z = inner(ψ', plaquette_mpo(p, sites, o), ψ)
        abs(imag(z)) < imag_tol ||
            @warn "String $k has non-negligible imaginary part" imag(z)
        vals[k] = plaquette_sign(o) * real(z)
    end
    return vals
end


# ------- Energy & variance ------------------------------------------------------------------------
"""
    measure_energy(ψ, H) -> Float64

Energy expectation `⟨ψ|H|ψ⟩` (real part). Cheap — one MPO–MPS contraction —
safe to call every sweep.
"""
measure_energy(ψ::MPS, H::MPO) = real(inner(ψ', H, ψ))


"""
    measure_variance(ψ, H) -> Float64

Energy variance `⟨ψ|H²|ψ⟩ − ⟨ψ|H|ψ⟩²` (real part). Small when ψ is near an
eigenstate of H. Expensive (contracts H twice) — call only for the final
report, never inside the per-sweep loop.
"""
function measure_variance(ψ::MPS, H::MPO)
    E  = real(inner(ψ', H, ψ))
    H2 = real(inner(H, ψ, H, ψ))
    return H2 - E^2
end


# ------- Validators (geometry-agnostic; take a geometry context) ----------------------------------
"""
    validate_circuit(circuit_gates, ψ_initial, geom; cutoff = 1e-6) -> NamedTuple

Apply `circuit_gates` to `ψ_initial`, then measure per-plaquette ⟨Wp⟩,
energy `E_opt = ⟨ψ|geom.H|ψ⟩`, and — when the geometry carries loop strings —
the loop expectations ⟨L⟩. Variance is opt-in via `measure_variance`.

Returns `(; ψ_opt, E_opt, wp_opt, lp_opt, plaquettes)`; `lp_opt` is empty for
geometries without loops.
"""
function validate_circuit(circuit_gates, ψ_initial::MPS, geom; cutoff::Real = 1e-6)
    ψ_opt = deepcopy(ψ_initial)
    for layer in circuit_gates
        ψ_opt = apply(layer, ψ_opt; cutoff)
    end
    normalize!(ψ_opt)

    sites  = siteinds(ψ_opt)
    wp_opt = measure_plaquettes(ψ_opt, sites, geom.plaquettes, geom.plaquette_ops)
    lp_opt = haskey(geom, :loops) ?
        measure_plaquettes(ψ_opt, sites, geom.loops, geom.loop_ops) : Float64[]
    E_opt  = measure_energy(ψ_opt, geom.H)
    return (; ψ_opt, E_opt, wp_opt, lp_opt, plaquettes = geom.plaquettes)
end


"""
    validate_reference(ψ_T, geom) -> NamedTuple

Per-plaquette ⟨Wp⟩, energy `E_target`, and (when the geometry carries loop
strings) ⟨L⟩ on the reference MPS. Variance is opt-in via `measure_variance`.

Returns `(; E_target, wp_target, lp_target, plaquettes)`; `lp_target` is
empty for geometries without loops.
"""
function validate_reference(ψ_T::MPS, geom)
    sites     = siteinds(ψ_T)
    wp_target = measure_plaquettes(ψ_T, sites, geom.plaquettes, geom.plaquette_ops)
    lp_target = haskey(geom, :loops) ?
        measure_plaquettes(ψ_T, sites, geom.loops, geom.loop_ops) : Float64[]
    E_target  = measure_energy(ψ_T, geom.H)
    return (; E_target, wp_target, lp_target, plaquettes = geom.plaquettes)
end