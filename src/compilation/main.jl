# 8/10/2026
# Compile a Kitaev cylinder ground state (WIDTH = 3 or 4) with a BOND-ONLY
# flux-sector-preserving Pauli-rotation circuit (no single-qubit layer):
#   (1) project the |↑…↑⟩ product state into the target's flux sector with
#       ∏ₚ (1 + sₚWₚ)/2, sₚ = sign⟨Wₚ⟩_target  →  all sₚ⟨Wₚ⟩ = +1 exactly;
#   (1b) project onto the target's non-contractible-loop sector with
#       (1 + sL)/2, L = Y₁…Y_2Ny;
#   (1c) project onto the target's edge-pairing sectors (PAIRING_STRINGS:
#       endpoint-rule strings pairing the boundary zero modes bʸ in disjoint
#       pairs — exact symmetries of H). The bond-only circuit conserves the
#       full commutant {Wₚ, L, pairing charges}; the projections match every
#       one of them to the target, so no fidelity ceiling remains, and the
#       frozen zero-mode sector shrinks the reachable Gaussian manifold:
#         width 3:  m = 12 modes (dim 132), 33 bonds/block → 3 blocks = 75%
#         width 4:  m = 16 modes (dim 240), 44 bonds/block → 3 blocks = 55%
#       (the width-4 run reached F = 0.994 at three blocks; the singles
#       ansatz at 41% stalled at F = 0.494).
#       The pairing basis is a SPONTANEOUS choice of the degenerate target —
#       each width's table below was MEASURED on its stored target (definite
#       ±1 on every string); projecting onto any other basis would cap F.
#   (2) act with one-parameter rotations e^{-iθ σᵅσᵅ} on the α-flavor bonds —
#       each is quadratic in Majoranas, so the in-sector circuit is Gaussian;
#   (3) optimize every angle analytically from its cached environment
#       (θ* = atan(A, B) — each gate is a one-parameter rotation);
#   (4) grow the circuit dynamically: optimize n_init blocks for nsweeps sweeps,
#       then append one identity block at a time up to n_total, re-optimizing
#       after each append (identity initialization at full depth is prone to
#       local maxima — an equal GS/excited superposition pinned at F = 1/√2).
# PAIRING_STRINGS is hard-coded per width below; everything else (plaquettes,
# loop, two-qubit layout, output naming) is derived from the model dimensions.


using ITensors, ITensorMPS
using LinearAlgebra
using HDF5
using Printf
using TimerOutputs


include("compute_cost_function.jl")           # from-scratch fidelity (debug cross-checks)
include("validation.jl")                      # validate_circuit / validate_reference, cluster_energy_mpo, honeycomb_lattice_Cstyle
include("cached_environment.jl")              # dagger_layer, build_psi_right (layer-level environment helpers)
include("cached_environment_nonlocal.jl")     # *_nl cache for non-adjacent, disjoint-interval gate layers
include("../helper/blas_setup.jl")            # platform BLAS backend + setup_threads!



# ------- Set up parameters for multithreading and parallelization -----------------------------------
setup_threads!(blocksparse = false)


# ------- Profiling -----------------------------------------------------------------------------------
const time_machine = TimerOutput()



# ------- Hyperparameters ----------------------------------------------------------------------------
const cutoff            = 1e-4
const projection_cutoff = 1e-12
const validation_cutoff = 1e-8
const default_iters     = 35
const min_iters         = 2
const stop_criteria     = 1e-5
const nsweeps           = 200                  # sweeps per growth stage
const n_init            = 2                    # blocks optimized from the start
const n_total           = 3                    # final block count; n_init == n_total → no growth


# ------- Cluster geometry + target ------------------------------------------------------------------
# One switch selects the cylinder: model, target file, and the target's
# MEASURED edge-pairing basis (endpoint rule: interior sites take the third
# flavor of their two path bonds, endpoints the flavor that is neither the
# link flavor nor the site's missing flavor). All strings are exact
# symmetries of H, definite (±1 to 1e-10) on their stored target.
const WIDTH = 3          # cylinder width Ny: 3 (N = 24) | 4 (N = 32)

if WIDTH == 3
    const model = (; Nx = 8, Ny = 3, Jx = 1.0, Jy = 1.0, Jz = 1.0, κ = -0.2,
                     yperiodic = true, pauli = true)
    const target_mps_path = joinpath(@__DIR__, "data",
        "cluster_n24_kappa-0.2_periodic.h5")
    # 6 zero modes (3 per edge, odd): one local pair per edge + the forced
    # cross-lattice pair (5,20). Measured: -1, +1, -1.
    const PAIRING_STRINGS = ((["Z", "Y", "X"], [1, 2, 3]),
                             (["X", "Y", "Z"], [22, 23, 24]),
                             (["Z", "Z", "Z", "Z", "Z", "Z", "X", "Y", "Y", "X"],
                              [5, 6, 11, 12, 17, 18, 23, 22, 21, 20]))
elseif WIDTH == 4
    const model = (; Nx = 8, Ny = 4, Jx = 1.0, Jy = 1.0, Jz = 1.0, κ = -0.3,
                     yperiodic = true, pauli = true)
    const target_mps_path = joinpath(@__DIR__, "data",
        "cluster_n32_kappa-0.3_periodic.h5")
    # 8 zero modes (4 per edge): all paired locally. Measured: all -1.
    const PAIRING_STRINGS = ((["Z", "Y", "X"], [1, 2, 3]),
                             (["Z", "Y", "X"], [5, 6, 7]),
                             (["X", "Y", "Z"], [26, 27, 28]),
                             (["X", "Y", "Z"], [30, 31, 32]))
else
    error("WIDTH must be 3 or 4, got $WIDTH")
end
const N = model.Nx * model.Ny


    
# ------- Gate library: one-parameter Pauli rotations  e^{-iθ P} = cos θ · I − i sin θ · P -----------
# with P = σᵅ (single site) or σᵅ⊗σᵅ (two sites, any distance). P² = I, so the
# rotation is unitary for every θ, and every gate commutes with all plaquette Wₚ.
const PAULI = Dict(
    "X" => ComplexF64[0 1; 1 0],
    "Y" => ComplexF64[0 -im; im 0],
    "Z" => ComplexF64[1 0; 0 -1],
)


"""
    pauli_itensors(sites, p, flavor) -> (Id, P)

Identity and Pauli-product ITensors on the site(s) `p` (length 1 or 2), with
legs `(s', s)` per site. Every rotation gate is `cos θ · Id − i sin θ · P`, and
the analytic update needs exactly these two contractions with the environment.
"""
function pauli_itensors(sites, p, flavor::String)
    id2 = Matrix{ComplexF64}(I, 2, 2)
    if length(p) == 1
        s = sites[p[1]]
        return ITensor(id2, s', s), ITensor(PAULI[flavor], s', s)
    else
        si, sj = sites[p[1]], sites[p[2]]
        Id = ITensor(id2, si', si) * ITensor(id2, sj', sj)
        P  = ITensor(PAULI[flavor], si', si) * ITensor(PAULI[flavor], sj', sj)
        return Id, P
    end
end


"""
    update_rotation_from_env(E_T, sites, p, flavor) -> (gate, value)

Analytic optimum of the one-parameter rotation on site(s) `p`: with
`B = Re((E_T · Id)[1])` and `A = Im((E_T · P)[1])`, the objective is
`f(θ) = Re((E_T · G(θ))[1]) = B cos θ + A sin θ`, maximized at `θ* = atan(A, B)`
with value `hypot(A, B)`. Returns the updated gate `cos θ*·Id − i sin θ*·P` and
`f(θ*)`. Coefficients come from direct contractions with the same Id/P tensors
the gate is built from, so no operator-convention bookkeeping is involved.
"""
function update_rotation_from_env(E_T, sites, p, flavor::String)
    Id, P = pauli_itensors(sites, p, flavor)
    B = real((E_T * Id)[1])
    A = imag((E_T * P)[1])
    θ = atan(A, B)
    return cos(θ) * Id - im * sin(θ) * P, hypot(A, B)
end


"""
    optimize_layer_nonlocal!(gates, idx_pairs, flavors, ψ_left, ψ_right, sites, N;
                             default_iters, stop_criteria, min_iters, layer_idx) -> Float64

Optimize one layer of one-parameter Pauli rotations by alternating left→right /
right→left sub-sweeps over the nonlocal cached environments (gate intervals must
be sorted and disjoint — asserted). Every update is the analytic optimum, so the
objective is non-decreasing. The k = 1 readout of the last right→left sub-sweep
equals the global overlap Re⟨ψ_T|U|ψ₀⟩ (up to the truncations in ψ_left/ψ_right);
early stopping compares consecutive readouts once `iter_idx > min_iters`.
Mutates `gates` in place; returns the final readout.
"""
function optimize_layer_nonlocal!(gates, idx_pairs, flavors, ψ_left, ψ_right, sites, N;
                                  default_iters::Int, stop_criteria::Real,
                                  min_iters::Int, layer_idx::Int)
    check_layer_intervals_nl(idx_pairs)
    M = length(idx_pairs)

    ups = Vector{ITensor}(undef, M)
    dns = Vector{ITensor}(undef, M)
    init_ups_left_nl!(ups, ψ_left, ψ_right, idx_pairs)
    precompute_dns_nl!(dns, ψ_left, ψ_right, gates, idx_pairs, N)

    fidelity₁ = fidelity₂ = 0.0
    for iter_idx in 1:default_iters
        # forward sub-sweep
        for k in 1:M
            E_T = build_env_nl(ups, dns, ψ_left, ψ_right, k, idx_pairs)
            gates[k], _ = update_rotation_from_env(E_T, sites, idx_pairs[k], flavors[k])
            k < M && extend_ups_nl!(ups, ψ_left, ψ_right, gates, idx_pairs, k)
        end

        # backward sub-sweep; k = 1 readout = global overlap
        for k in M:-1:1
            E_T = build_env_nl(ups, dns, ψ_left, ψ_right, k, idx_pairs)
            gates[k], val = update_rotation_from_env(E_T, sites, idx_pairs[k], flavors[k])
            if k > 1
                contract_dns_from_right_nl!(dns, ψ_left, ψ_right, gates, idx_pairs, k)
            else
                fidelity₂ = val
                # DEBUG: env readout must equal a direct layer contraction to machine precision
                # fidelity_direct = real(inner(ψ_right, apply(gates, ψ_left; cutoff = 1e-15)))
                # abs(fidelity₂ - fidelity_direct) < 1e-10 || @warn "nl env readout mismatch" layer_idx fidelity₂ fidelity_direct
            end
        end

        if iter_idx > min_iters && abs(fidelity₂ - fidelity₁) < stop_criteria
            break
        end
        fidelity₁ = fidelity₂
    end

    return fidelity₂
end


"""
    bond_flavor(b, Ny) -> "X" | "Y" | "Z"

Kitaev flavor of lattice bond `b` under the C-style dispatch — the single
source of truth, identical to `cluster_energy_mpo`: even-column bonds are Y,
and odd columns split into X (vertical neighbours, `s2 - s1 == 1`) and Z (the
rest, including the cylinder's y-wrap bonds).
"""
function bond_flavor(b, Ny)
    xcoord = 2 * div(b.s1 - 1, 2 * Ny) + (iseven(b.s1) ? 2 : 1)
    return iseven(xcoord) ? "Y" : (b.s2 - b.s1 == 1 ? "X" : "Z")
end


"""
    flavor_layers(bonds, Ny) -> Vector{Tuple{String, Vector{Vector{Int}}}}

Partition the lattice bonds into layers of same-flavor two-qubit gates whose
site intervals are sorted and pairwise disjoint (the nonlocal-cache layer
requirement, asserted by `check_layer_intervals_nl`). Bonds of each flavor are
sorted by first site and placed first-fit into the earliest compatible layer;
within each flavor, layers are then ordered by descending gate count. This
reproduces the hand-written cylinder layout in its exact historic order
(X, Z-normal, Z-wrap, then the Y layers — three at width 3, four at width 4 —
sweep trajectories match the original runs); width-3 open boundaries yield
1 X, 1 Z, 3 Y layers (29 bonds).
"""
function flavor_layers(bonds, Ny)
    layers = Vector{Tuple{String, Vector{Vector{Int}}}}()
    for f in ("X", "Z", "Y")
        flayers = Vector{Vector{Vector{Int}}}()
        for p in sort([sort([b.s1, b.s2]) for b in bonds if bond_flavor(b, Ny) == f])
            k = findfirst(l -> l[end][2] < p[1], flayers)
            k === nothing ? push!(flayers, [p]) : push!(flayers[k], p)
        end
        sort!(flayers; by = length, rev = true)   # larger layers first (historic order)
        for prs in flayers
            push!(layers, (f, prs))
        end
    end
    return layers
end


"""
    append_block!(input_pairs, gate_flavors, circuit_gates, sites, tq_layers)

Append one identity-initialized block of two-qubit bond layers to the circuit
containers. Appending at the end places the new block closest to the target,
so a freshly added identity block leaves the current fidelity unchanged and
each growth stage starts from the previous stage's optimum.
"""
function append_block!(input_pairs, gate_flavors, circuit_gates, sites, tq_layers)
    for (f, prs) in tq_layers
        push!(input_pairs,   prs)
        push!(gate_flavors,  fill(f, length(prs)))
        push!(circuit_gates, [pauli_itensors(sites, p, f)[1] for p in prs])
    end
end


"""
    gate_angle(g, sites, p, flavor) -> θ

Recover the rotation angle of gate `g = cos θ·Id − i sin θ·P` by trace
projection onto the same Id/P tensors the gate was built from. Saved with the
results so a converged circuit can be reused as a warm start.
"""
function gate_angle(g, sites, p, flavor::String)
    Id, P = pauli_itensors(sites, p, flavor)
    d = 2.0^length(p)
    return atan(-imag((g * dag(P))[1]) / d, real((g * dag(Id))[1]) / d)
end



let
    t_start = time_ns()
    println(repeat("-", 150))
    println("Flux-sector-preserving Pauli-rotation compilation of the Kitaev cluster ground state")
    println(repeat("-", 150), "\n")


    # ------- Load the target ground-state MPS ---------------------------------------------------------
    isfile(target_mps_path) || error("target MPS not found: $target_mps_path")
    ψ_T, sites = h5open(target_mps_path, "r") do file
        ψ = read(file, "psi", MPS)
        return ψ, siteinds(ψ)
    end
    @assert length(sites) == N "loaded $(length(sites)) sites but model implies N=$N"
    @info "Loaded target MPS" path=target_mps_path N=length(sites) maxlinkdim=maxlinkdim(ψ_T)


    # ------- (1) Project |↑…↑⟩ into the target's flux sector: ψ_init = ∏ₚ (1 + sₚWₚ)/2 |↑…↑⟩ ---------
    # Plaquette table generated from the C-style anchor rule for any cylinder
    # width: anchor (x, y) with the bottom row at y - 1, wrapped at y = 1; the
    # fixed op pattern is the outward flavor at each cycle position, uniform
    # for every width. Rows whose six edges are not all lattice bonds — the
    # seam hexagons, absent for open boundaries — are filtered out below.
    site_at(x, y) = 2 * model.Ny * div(x - 1, 2) + 2 * (y - 1) + (isodd(x) ? 1 : 2)
    plaquette_rows = Matrix{Int}[]
    for x in 1:2:(model.Nx - 3), y in 1:model.Ny
        yb = y == 1 ? model.Ny : y - 1
        push!(plaquette_rows, [site_at(x, y) site_at(x + 1, y) site_at(x + 2, y) site_at(x + 3, yb) site_at(x + 2, yb) site_at(x + 1, yb)])
    end
    master_plaquettes = vcat(plaquette_rows...)
    plaquette_ops = ["Y", "Z", "X", "Y", "Z", "X"]

    bonds   = honeycomb_lattice_Cstyle(model.Nx, model.Ny; yperiodic = model.yperiodic)
    bondset = Set(minmax(b.s1, b.s2) for b in bonds)
    valid_rows = [r for r in 1:size(master_plaquettes, 1)
                  if all(minmax(master_plaquettes[r, k], master_plaquettes[r, mod1(k + 1, 6)]) in bondset
                         for k in 1:6)]
    plaquettes = master_plaquettes[valid_rows, :]
    @info "Plaquettes" count = size(plaquettes, 1) yperiodic = model.yperiodic

    # Projection signs follow the TARGET's flux sector (all +1 for the stored
    # flux-free ground states; ±1-robust for future flux-pinned targets).
    wp_sign = map(1:size(plaquettes, 1)) do r
        os = OpSum()
        add!(os, 1.0, collect(Iterators.flatten(zip(plaquette_ops, plaquettes[r, :])))...)
        w = real(inner(ψ_T', MPO(os, sites), ψ_T))
        abs(w) > 1 - 1e-6 || @warn "target flux is not definite on plaquette" r w
        sign(w)
    end

    ψ_init = MPS(sites, fill("Up", N))
    @timeit time_machine "flux projection" for r in 1:size(plaquettes, 1)
        os = OpSum()
        os += 0.5, "Id", 1
        add!(os, 0.5 * wp_sign[r], collect(Iterators.flatten(zip(plaquette_ops, plaquettes[r, :])))...)
        ψ_init = apply(MPO(os, sites), ψ_init; cutoff = projection_cutoff)
        normalize!(ψ_init)
    end

    
    
    # ------- (1b) Cylinder only: project onto the target's loop sector, L = Y₁…Y_2Ny ------------------
    # On the cylinder the circuit conserves the non-contractible loop L, the
    # flux-projected |↑…↑⟩ has ⟨L⟩ = 0, and a definite-L target then caps the
    # reachable overlap at 1/√2. With open boundaries the boundary single-qubit
    # gates break L (the commutant is exactly {Wₚ}) — no projection is needed.
    if model.yperiodic
        @timeit time_machine "loop projection" begin
        L_mpo = let os = OpSum()
            add!(os, 1.0, collect(Iterators.flatten(zip(fill("Y", 2 * model.Ny), 1:2 * model.Ny)))...)
            MPO(os, sites)
        end
        loop_target = real(inner(ψ_T', L_mpo, ψ_T))
        abs(loop_target) > 1 - 1e-6 || @warn "target loop sector is not definite" loop_target
        s = sign(loop_target)
        ψ_init = normalize!(+(0.5 * ψ_init, (0.5 * s) * apply(L_mpo, ψ_init; cutoff = projection_cutoff);
                              cutoff = projection_cutoff))
        Lval = real(inner(ψ_init', L_mpo, ψ_init))
        abs(Lval - s) < 1e-6 || error("loop-sector projection failed: ⟨L⟩ = $Lval")
        @info "Loop-sector-projected initial state" loop_target
        end
    end


    # ------- (1c) Project onto the target's edge-pairing sectors --------------------------------------
    # The bond-only ansatz conserves the pairing charges, so the initial state
    # must carry the target's eigenvalues; the projection also freezes the
    # zero-mode sector (see header). Signs are measured from the target, so
    # the step is robust to future targets in other pairing sectors.
    pairing_signs = Float64[]
    @timeit time_machine "pairing projections" for (ops, ss) in PAIRING_STRINGS
        T_mpo = let os = OpSum()
            add!(os, 1.0, collect(Iterators.flatten(zip(ops, ss)))...)
            MPO(os, sites)
        end
        t = real(inner(ψ_T', T_mpo, ψ_T))
        abs(t) > 1 - 1e-6 || @warn "target pairing sector is not definite" ops ss t
        s = sign(t)
        push!(pairing_signs, s)
        ψ_init = normalize!(+(0.5 * ψ_init, (0.5 * s) * apply(T_mpo, ψ_init; cutoff = projection_cutoff);
                              cutoff = projection_cutoff))
        Tval = real(inner(ψ_init', T_mpo, ψ_init))
        abs(Tval - s) < 1e-6 || error("pairing-sector projection failed: ⟨T⟩ = $Tval (target sign $s)")
    end
    @info "Pairing-sector-projected initial state" signs = pairing_signs χ = maxlinkdim(ψ_init)


    # Sector check: sₚ⟨Wₚ⟩ must be +1 on the projected state for every plaquette.
    wp_init = map(1:size(plaquettes, 1)) do r
        os = OpSum()
        add!(os, 1.0, collect(Iterators.flatten(zip(plaquette_ops, plaquettes[r, :])))...)
        real(inner(ψ_init', MPO(os, sites), ψ_init))
    end
    minimum(wp_sign .* wp_init) > 1 - 1e-8 ||
        error("projection failed: min sₚ⟨Wₚ⟩ = $(minimum(wp_sign .* wp_init))")
    F_init = abs(inner(ψ_T, ψ_init))
    @info "Projected initial state" min_Wp=minimum(wp_sign .* wp_init) χ=maxlinkdim(ψ_init) fidelity_vs_target=F_init


    
    # ------- Hamiltonian context for energy / ⟨Wₚ⟩ / ⟨L⟩ validation ----------------------------------
    # Built from the same generated tables as the projection; loops are the
    # all-Y rings, one per column pair — conserved by the cylinder circuit,
    # degeneracy probes under open boundaries.
    geom = (; H = cluster_energy_mpo(sites; model...),
              plaquettes    = [plaquettes[r, :] for r in 1:size(plaquettes, 1)],
              plaquette_ops = [plaquette_ops for _ in 1:size(plaquettes, 1)],
              loops         = [collect(2 * model.Ny * (p - 1) .+ (1:2 * model.Ny)) for p in 1:(model.Nx ÷ 2)],
              loop_ops      = [fill("Y", 2 * model.Ny) for _ in 1:(model.Nx ÷ 2)])


    
    # ------- (2) Flux-preserving gate layout (derived from the lattice bond list) ---------------------
    # Bond-only ansatz: two-qubit rotations carrying the flavor of their bond,
    # derived from `bonds` so the layout adapts to the geometry automatically.
    # The boundary single-qubit rotations of the previous ansatz are superseded
    # by the pairing-sector projection (1c) — the circuit now conserves the
    # pairing charges instead of steering them.
    tq_layers = flavor_layers(bonds, model.Ny)
    sum(length(prs) for (_, prs) in tq_layers) == length(bonds) ||
        error("two-qubit layers do not cover every lattice bond exactly once")
    @info "Gate layout" two_qubit_layers=length(tq_layers) bonds=length(bonds)


    # Build the starting circuit: n_init blocks of two-qubit bond layers,
    # every gate initialized at θ = 0 (identity).
    input_pairs   = Vector{Vector{Vector{Int}}}()
    gate_flavors  = Vector{Vector{String}}()
    circuit_gates = Vector{Vector{ITensor}}()
    for _ in 1:n_init
        append_block!(input_pairs, gate_flavors, circuit_gates, sites, tq_layers)
    end
    @info "Circuit layout" blocks=n_init layers=length(circuit_gates) gates=sum(length.(circuit_gates))


    # ------- (3) Optimize all angles analytically, growing the circuit block by block -----------------
    # Each growth stage appends one identity block (fidelity is continuous across
    # the append) and re-optimizes; warm-starting from the previous stage's optimum
    # avoids the local maxima that identity initialization hits at full depth.
    cost_function   = Float64[]
    energy_trace    = Float64[]
    plaquette_trace = Vector{Float64}[]
    loop_trace      = Vector{Float64}[]
    block_trace     = Int[]                   # blocks active at each recorded sweep
    stage_fidelity  = Float64[]               # fidelity at the end of each growth stage

    for stage_blocks in n_init:n_total
        if stage_blocks > n_init
            append_block!(input_pairs, gate_flavors, circuit_gates, sites, tq_layers)
        end
        nlayers = length(circuit_gates)
        @info "Growth stage" blocks=stage_blocks layers=nlayers gates=sum(length.(circuit_gates))

        ψ_left_collection    = Vector{MPS}(undef, nlayers)
        ψ_left_collection[1] = ψ_init
        ψ_right_collection   = @timeit time_machine "build ψ_right" build_psi_right(circuit_gates, ψ_T; cutoff)

        for iteration in 1:nsweeps
            println(repeat("-", 150))
            @printf "Block %d/%d  Sweep %d/%d\n\n" stage_blocks n_total iteration nsweeps

            fidelity_sweep = 0.0
            # Forward pass: optimize layers bottom → top, maintaining ψ_left on the fly
            @timeit time_machine "sweep: forward pass" for layer_idx in 1:nlayers
                optimize_layer_nonlocal!(circuit_gates[layer_idx], input_pairs[layer_idx],
                    gate_flavors[layer_idx], ψ_left_collection[layer_idx],
                    ψ_right_collection[layer_idx], sites, N;
                    default_iters, stop_criteria, min_iters, layer_idx)
                if layer_idx < nlayers
                    ψ_left_collection[layer_idx + 1] =
                        normalize!(apply(circuit_gates[layer_idx], ψ_left_collection[layer_idx]; cutoff))
                end
            end

            # Backward pass: optimize layers top → bottom, maintaining ψ_right on the fly
            @timeit time_machine "sweep: backward pass" for layer_idx in nlayers:-1:1
                fid = optimize_layer_nonlocal!(circuit_gates[layer_idx], input_pairs[layer_idx],
                    gate_flavors[layer_idx], ψ_left_collection[layer_idx],
                    ψ_right_collection[layer_idx], sites, N;
                    default_iters, stop_criteria, min_iters, layer_idx)
                layer_idx > 1 && (ψ_right_collection[layer_idx - 1] =
                    normalize!(apply(dagger_layer(circuit_gates[layer_idx]), ψ_right_collection[layer_idx]; cutoff)))
                layer_idx == 1 && (fidelity_sweep = fid)
            end

            push!(cost_function, fidelity_sweep)
            push!(block_trace, stage_blocks)
            # fidelity_sweep = compute_cost_function_multi_layers(ψ_init, ψ_T, circuit_gates, cutoff)   # DEBUG cross-check

            en = @timeit time_machine "per-sweep validation" validate_circuit(circuit_gates, ψ_init, geom; cutoff)
            push!(energy_trace,    en.E_opt)
            push!(plaquette_trace, en.wp_opt)
            push!(loop_trace,      en.lp_opt)

            @printf "-------- block %d/%d  sweep %d/%d done  Fidelity = %+.6f  Energy = %+.6f  min⟨Wp⟩ = %+.8f  ⟨L⟩ = %s\n" stage_blocks n_total iteration nsweeps fidelity_sweep en.E_opt minimum(en.wp_opt) join(
                (@sprintf("%+.6f", x) for x in en.lp_opt), " ")
            println(repeat("-", 150), "\n")
        end
        push!(stage_fidelity, cost_function[end])
    end


    # ------- Final validation against the target ------------------------------------------------------
    compiled = @timeit time_machine "final validation" validate_circuit(circuit_gates, ψ_init, geom; cutoff = validation_cutoff)
    target   = @timeit time_machine "final validation" validate_reference(ψ_T, geom)
    F_final  = abs(inner(ψ_T, compiled.ψ_opt))
    ΔE       = compiled.E_opt - target.E_target

    # Cross-check ⟨ψ_T|H|ψ_T⟩ against the DMRG's stored eigenvalue: agreement to
    # ~1e-6 for well-converged targets; any Hamiltonian-term or convention
    # mismatch shows up far above that (this check caught the OBC seam-wedge bug).
    E0_stored = h5open(target_mps_path, "r") do f; haskey(f, "E0") ? read(f, "E0") : nothing; end
    if E0_stored !== nothing
        ΔE0 = abs(target.E_target - E0_stored)
        ΔE0 < 1e-6 || @warn "Computed target energy disagrees with stored E0" E0_stored target.E_target ΔE0
    end


    # ------- Save --------------------------------------------------------------------------------------
    outdir = joinpath(@__DIR__, "data", "kitaev"); mkpath(outdir)
    output_filename = joinpath(outdir,
        "pauli_cluster_$(model.Nx)x$(model.Ny)_kappa$(model.κ)_periodic_l$(n_total)_paired.h5")
    h5open(output_filename, "w") do file
        write(file, "psi_opt",         compiled.ψ_opt)
        write(file, "psi_init",        ψ_init)
        write(file, "cost_function",   cost_function)
        write(file, "energy_trace",    energy_trace)
        write(file, "plaquette_trace", Matrix(reduce(hcat, plaquette_trace)'))
        write(file, "Wp_opt",          compiled.wp_opt)
        write(file, "Wp_target",       target.wp_target)
        write(file, "L_opt",           compiled.lp_opt)
        write(file, "L_target",        target.lp_target)
        write(file, "loop_trace",      Matrix(reduce(hcat, loop_trace)'))
        write(file, "fidelity",        F_final)
        write(file, "fidelity_init",   F_init)
        write(file, "block_trace",     block_trace)
        write(file, "stage_fidelity",  stage_fidelity)
        write(file, "gate_angles",     [gate_angle(circuit_gates[l][k], sites, input_pairs[l][k], gate_flavors[l][k])
                                        for l in eachindex(circuit_gates) for k in eachindex(circuit_gates[l])])
        write(file, "pairing_signs",   pairing_signs)
    end
    @info "Saved results" output_filename


    # ------- Report ------------------------------------------------------------------------------------
    section(title) = (println("\n", repeat("-", 100)); println(title); println(repeat("-", 100)))

    section("Energy and fidelity")
    @printf "  %-10s E  = %+.8f\n" "target"   target.E_target
    @printf "  %-10s E  = %+.8f\n" "compiled" compiled.E_opt
    @printf "  %-10s ΔE = %+.8f\n" "gap"      ΔE
    @printf "  fidelity:  initial (projected) = %.8f   compiled = %.8f\n" F_init F_final

    section("⟨L⟩ on each ring (topological sector)")
    for (label, lp) in (("optimized", compiled.lp_opt), ("target", target.lp_target))
        @printf "  %-12s %s\n" label join((@sprintf("%+.8f", x) for x in lp), "  ")
    end

    section("⟨Wp⟩ on each plaquette")
    for (label, wp) in (("optimized", compiled.wp_opt), ("target", target.wp_target))
        @printf "  %-12s %s\n" label join((@sprintf("%+.8f", x) for x in wp), "  ")
    end


    # ------- Timing ------------------------------------------------------------------------------------
    section("Timing")
    print_timer(time_machine; allocations = false, sortby = :time)
    @printf "\n\n  Total wall time: %.1f s\n" (time_ns() - t_start) / 1e9


    return
end                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     