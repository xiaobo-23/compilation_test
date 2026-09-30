# 11/1/2025
# From-scratch fidelity of a multi-layer circuit: apply every layer to the ket and take the
# overlap. Slower than the cached-environment readout but fully independent of it — main.jl's
# commented DEBUG cross-check calls this to validate the cached sweep fidelity.

using ITensors
using ITensorMPS


"""
    compute_cost_function_multi_layers(psi_ket, psi_bra, input_gates,
                                       input_cutoff = 1e-12) -> Float64

Overlap `Re⟨ψ_bra| U |ψ_ket⟩` with `U` the circuit `input_gates` applied layer
by layer (each layer a `Vector{ITensor}`), normalizing once at the end.
"""
function compute_cost_function_multi_layers(psi_ket::MPS, psi_bra::MPS,
                                            input_gates::Vector{Vector{ITensor}},
                                            input_cutoff::Float64 = 1e-12)
    for layer in input_gates
        psi_ket = apply(layer, psi_ket; cutoff = input_cutoff)
    end
    normalize!(psi_ket)
    return real(inner(psi_bra, psi_ket))
end
