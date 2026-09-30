# 6/1/2026
# Kitaev honeycomb cluster Hamiltonian as an MPO (spin-1/2 & Pauli conventions).

using ITensors
using ITensorMPS



# ------- Set up the Hamiltonian on a cluster as an MPO --------------------------------------------
"""
    cluster_energy_mpo(sites; Nx, Ny, Jx = 1.0, Jy = 1.0, Jz = 1.0, κ = 0.0,
                       yperiodic::Bool = true, pauli::Bool = false) -> MPO

Build the Kitaev honeycomb Hamiltonian as an MPO on the C-style site labelling:

    H = -Jx Σ σˣᵢ σˣⱼ  - Jy Σ σʸᵢ σʸⱼ  - Jz Σ σᶻᵢ σᶻⱼ
        + κ  Σ σᵅᵢ σᵝⱼ σᵞₖ      (three-spin "wedge" terms)

Operator convention: `pauli = false` (default) emits spin-1/2 operators
(`Sx, Sy, Sz` — the legacy convention of older stored targets); `pauli = true`
emits Pauli operators (`X, Y, Z` — required for the shipped targets, e.g.
`data/cluster_n24_kappa-0.2_periodic.h5`). The two differ by a factor of 4 per
bond and 8 per wedge term.

Bond and wedge dispatch matches the DMRG driver that produced the stored
targets (and is mirrored by `bond_flavor` in main.jl). Set `κ = 0.0` to skip
the wedge construction; `yperiodic = false` builds the fully open cluster.
"""
function cluster_energy_mpo(sites; Nx::Integer, Ny::Integer,
                    Jx::Real = 1.0, Jy::Real = 1.0, Jz::Real = 1.0,
                    κ::Real  = 0.0, yperiodic::Bool = true, pauli::Bool = false)

    @assert length(sites) == Nx * Ny "sites length $(length(sites)) ≠ Nx*Ny = $(Nx*Ny)"

    # Operator names for the chosen convention (used in both bond and wedge terms)
    OX, OY, OZ = pauli ? ("X", "Y", "Z") : ("Sx", "Sy", "Sz")

    # Set up the Hamiltonian with two-body and three-spin terms as an MPO
    os = OpSum()


    # ------- two-body Kitaev bonds --------------------------------------------------------------
    bonds = honeycomb_lattice_Cstyle(Nx, Ny; yperiodic)
    xbond, ybond, zbond = 0, 0, 0


    for b in bonds
        xcoord = 2 * div(b.s1 - 1, 2 * Ny) + (iseven(b.s1) ? 2 : 1)

        if iseven(xcoord)
            os .+= -Jy, OY, b.s1, OY, b.s2
            # @show b.s1, b.s2, "Sy"
            ybond += 1
        elseif b.s2 - b.s1 == 1
            os .+= -Jx, OX, b.s1, OX, b.s2
            # @show b.s1, b.s2, "Sx"
            xbond += 1
        else
            os .+= -Jz, OZ, b.s1, OZ, b.s2
            # @show b.s1, b.s2, "Sz"
            zbond += 1
        end
    end
    xbond + ybond + zbond == length(bonds) || error("Setting up $(xbond + ybond + zbond) bonds instead of $(length(bonds)) inconsistent.")


    # ------- three-spin (wedge) terms -----------------------------------------------------------
    if abs(κ) > 1e-12
        wedge = honeycomb_Cstyle_wedge(Nx, Ny; yperiodic)
        count = 0
        
        for w in wedge 
            tmp = div(w.s2 - 1, 2 * Ny)
            x = 2 * tmp + mod(w.s2 - 1, 2) + 1

            if isodd(x)
                if w.s1 - w.s2 == 1 && w.s3 - w.s2 == 2 * Ny - 1
                    os .+= κ, OX, w.s1, OY, w.s2, OZ, w.s3
                    # @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
                    count += 1
                end

                if w.s3 - w.s2 == 1 && w.s2 - w.s1 == 1
                    os .+= κ, OZ, w.s1, OY, w.s2, OX, w.s3
                    # @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
                    count += 1
                end 

                if x != 1 && w.s2 - w.s1 == 2 * Ny - 1
                    if w.s3 - w.s2 == 1
                        os .+= κ, OY, w.s1, OZ, w.s2, OX, w.s3
                        # @show w.s1, w.s2, w.s3, "Sy", "Sz", "Sx"
                        count += 1  
                    else
                        os .+= κ, OY, w.s1, OX, w.s2, OZ, w.s3
                        # @show w.s1, w.s2, w.s3, "Sy", "Sx", "Sz"
                        count += 1  
                    end
                end
            else
                if w.s3 - w.s2 == w.s2 - w.s1 == 1
                    os .+= κ, OX, w.s1, OY, w.s2, OZ, w.s3
                    # @show w.s1, w.s2, w.s3, "Sx", "Sy", "Sz"
                    count += 1
                end

                if w.s2 - w.s3 == 1 && w.s2 - w.s1 == 2 * Ny - 1
                    os .+= κ, OZ, w.s1, OY, w.s2, OX, w.s3
                    # @show w.s1, w.s2, w.s3, "Sz", "Sy", "Sx"
                    count += 1
                end

                if x != Nx && w.s3 - w.s2 == 2 * Ny - 1
                    if w.s2 - w.s1 == 1
                        os .+= κ, OX, w.s1, OZ, w.s2, OY, w.s3
                        # @show w.s1, w.s2, w.s3, "Sx", "Sz", "Sy"
                        count += 1
                    else
                        os .+= κ, OZ, w.s1, OX, w.s2, OY, w.s3
                        # @show w.s1, w.s2, w.s3, "Sz", "Sx", "Sy"
                        count += 1
                    end
                end
            end
        end
        # @show count

        count == length(wedge) || error("Wedge dispatch covered $count of $(length(wedge)) wedges — bond classification is inconsistent.")
    end

    return MPO(os, sites)
end