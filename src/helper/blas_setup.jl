# 8/5/2026
# Shared CPU BLAS backend + threading setup for the compilation driver: the per-platform
# backend, chosen at load time and routed via libblastrampoline (Intel MKL on x86 clusters,
# Julia's default OpenBLAS on Apple silicon), plus the BLAS/ITensors thread configuration.

using ITensors
using LinearAlgebra

# Apple's Accelerate was evaluated and removed: its LAPACK segfaults on large complex
# SVDs (reproduced in TDVP at maxdim = 512 with a χ = 256 state), and a BLAS that
# corrupts memory in one regime cannot be trusted for silent correctness in others.
# Re-adding it is a two-line change here once a fixed macOS version is benchmarked
# and verified.
@static if !(Sys.isapple() && Sys.ARCH === :aarch64)
    using MKL
    const BLAS_BACKEND = "MKL"
else
    const BLAS_BACKEND = "OpenBLAS (Apple-silicon default)"
end


"""
    default_blas_threads() -> Int

BLAS thread count for this machine: `SLURM_CPUS_PER_TASK` under Slurm;
otherwise 4 on Apple silicon (measured on an M-series laptop: matmul scales
through 8 threads, but LAPACK's complex SVD — the kernel that dominates MPS
gate application — is fastest at 4 and degrades badly beyond, 2.2 s → 0.77 s
→ 7 s for 1 → 4 → 8 threads on zgesdd 1024²); otherwise min(8, logical cores).
"""
function default_blas_threads()
    haskey(ENV, "SLURM_CPUS_PER_TASK") && return parse(Int, ENV["SLURM_CPUS_PER_TASK"])
    Sys.isapple() && Sys.ARCH === :aarch64 && return 4
    return min(8, Sys.CPU_THREADS)
end


"""
    setup_threads!(; blas = default_blas_threads(), blocksparse = false)

Set the BLAS thread count and configure ITensors' threading. With
`blocksparse = true` and Julia running on more than one thread, dense Strided
threading is disabled (avoiding oversubscription against BLAS) and threaded
block-sparse contractions are enabled; otherwise block-sparse threading is
switched off and Strided is limited to one thread — the right choice without
conserved quantum numbers. Logs the resulting configuration.
"""
function setup_threads!(; blas::Int = default_blas_threads(), blocksparse::Bool = false)
    BLAS.set_num_threads(blas)
    if blocksparse && Threads.nthreads() > 1
        ITensors.Strided.disable_threads()
        ITensors.enable_threaded_blocksparse()
    else
        ITensors.Strided.set_num_threads(1)
        ITensors.enable_threaded_blocksparse(false)
    end
    @info "Threading" backend = BLAS_BACKEND blas_threads = BLAS.get_num_threads() julia_threads =
        Threads.nthreads() blocksparse_threaded = ITensors.using_threaded_blocksparse()
    return nothing
end
