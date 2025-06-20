// brain.metal  –  Monte-Carlo traversal with
//                 • global spike-budget
//                 • Hebbian STDP  (+/-)
//                 • homeostatic weight drift toward a target firing rate
//                 • reward-modulated plasticity (three-factor rule)

#include <metal_stdlib>
using namespace metal;

/* packed synapse record (identical layout on CPU) */
struct SynapsePacked { uint src, dst; float w, pad; };

/* ------------------------------------------------------------
   RNG helper: 32-bit xorshift -> [0,1) float                */
inline float rand01(uint s)
{
    s ^= s << 13;  s ^= s >> 17;  s ^= s << 5;
    return (float)(s & 0xFFFFFF) * (1.0f / 16777216.0f);
}

/* ------------- build-time knobs --------------------------- */
#define BASE_SCALE     0.8f      /* p = w² · BASE_SCALE                 */
#define REFRACTORY      2u       /* dst silent window [ticks]           */
#define WINDOW_PRE      5u       /* pre spike valid window              */
#define MAX_SPIKES    128u       /* global budget per kernel pass       */
#define CLOCK_INC       1u       /* add per synapse                      */

#define TARGET_RATE_HZ 1000.0f   /* homeostatic set-point               */
#define ETA_HOME      1.0e-6f    /* homeostasis learning rate           */
#define ETA_REWARD    1.0e-3f    /* reward modulation scale             */
#define ALPHA_RBAR    0.001f     /* EWMA for running reward average     */

/* ────────────────────────────────────────────────────────────────── *
 *  monte_carlo_traversal  –  unit-step per-TG clock                 *
 *  Only diff vs. original: lines ####-#### below.                   *
 * ────────────────────────────────────────────────────────────────── */
/* ============================================================ *
 *  monte_carlo_traversal  –  per-thread-group clock cache       *
 *  (same logic as the original; only lines marked ★ differ)     *
 * ============================================================ */
kernel void monte_carlo_traversal(
    device       SynapsePacked* syn   [[buffer(0)]],
    device atomic_uint*  lastF        [[buffer(1)]],
    device atomic_uint*  lastV        [[buffer(2)]],        /* unused */
    device atomic_uint*  clock        [[buffer(3)]],
    constant uint&       nSyn         [[buffer(4)]],
    constant uint&       tau_vis      [[buffer(5)]],
    constant uint&       tau_pre      [[buffer(6)]],
    constant float&      aLTP         [[buffer(7)]],
    constant float&      aLTD         [[buffer(8)]],
    constant float&      wMin         [[buffer(9)]],
    constant float&      wMax         [[buffer(10)]],
    device atomic_uint*  budget       [[buffer(11)]],
    device const float*  reward       [[buffer(12)]],       /* scalar 0/1 */
    device atomic_float* rBarA        [[buffer(13)]],       /* running avg */
    uint3                tPos         [[thread_position_in_threadgroup]],
    uint3                gPos         [[thread_position_in_grid]],
    uint3                tgSize       [[threads_per_threadgroup]])
{
    const uint tid = gPos.x;                 /* global 1-D index          */
    if (tid >= nSyn) return;

    /* ---------- fast per-TG clock cache --------------------------- */
    threadgroup uint tgNow = 0;                                     /* ★ */
    if (tPos.x == 0)                                            /* ★ */
        tgNow = atomic_load_explicit(clock, memory_order_relaxed);/* ★ */
    threadgroup_barrier(mem_flags::mem_threadgroup);            /* ★ */
    uint now = tgNow;                                           /* ★ */

    SynapsePacked s = syn[tid];

    /* ---------- gating ------------------------------------------- */
    uint lp = atomic_load_explicit(&lastF[s.src], memory_order_relaxed);
    if (now - lp > WINDOW_PRE) {
        if (tid == 0) atomic_fetch_add_explicit(clock, CLOCK_INC, memory_order_relaxed);
        return;
    }

    uint ld = atomic_load_explicit(&lastF[s.dst], memory_order_relaxed);
    if (now - ld <= REFRACTORY) {
        if (tid == 0) atomic_fetch_add_explicit(clock, CLOCK_INC, memory_order_relaxed);
        return;
    }

    if (atomic_load_explicit(budget, memory_order_relaxed) == 0u) {
        if (tid == 0) atomic_fetch_add_explicit(clock, CLOCK_INC, memory_order_relaxed);
        return;
    }

    /* probability -------------------------------------------------- */
    float p     = clamp(s.w * s.w * BASE_SCALE, 0.f, 1.f);
    bool  fired = (p > rand01(tid ^ now));

    /* global budget ------------------------------------------------ */
    if (fired) {
        uint old = atomic_fetch_sub_explicit(budget, 1u, memory_order_relaxed);
        if (old == 0u) fired = false;              /* lost race */
    }

    /* ---------------- plasticity --------------------------------- */
    float dW = fired ?  (aLTP * (1.f - s.w))
                     : (-aLTD * s.w);

    /* reward-modulated term (three-factor) ------------------------- */
    float R     = *reward;
    float rBar  = atomic_load_explicit(rBarA, memory_order_relaxed);
    dW         += ETA_REWARD * (R - rBar) * (fired ? 1.0f : 0.0f);

    /* EWMA for reward baseline – any one thread is fine ------------ */
    if (tid == 0) {
        float newBar = rBar + ALPHA_RBAR * (R - rBar);
        atomic_store_explicit(rBarA, newBar, memory_order_relaxed);
    }

    /* homeostatic drift toward TARGET_RATE_HZ ---------------------- */
    float isi   = float(now - ld);
    float estHz = isi > 0.f ? 1e6f / isi : 0.f;     /* tick = 1 µs */
    dW         += ETA_HOME * (TARGET_RATE_HZ - estHz) * s.w;

    /* clamp and write-back weight ---------------------------------- */
    s.w = clamp(s.w + dW, wMin, wMax);
    syn[tid] = s;

    /* update post-neuron last-fire if spike occurred --------------- */
    if (fired)
        atomic_store_explicit(&lastF[s.dst], now, memory_order_relaxed);

    /* one clock tick per kernel pass ------------------------------- */
    if (tid == 0) atomic_fetch_add_explicit(clock, CLOCK_INC, memory_order_relaxed);
}



/* ============================================================ */
kernel void renormalise_clock_and_times(
    device atomic_uint* lastF   [[buffer(0)]],
    device atomic_uint* lastVis [[buffer(1)]],
    device atomic_uint* clk     [[buffer(2)]],
    constant uint&      nNeur   [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint base = atomic_load_explicit(clk, memory_order_relaxed);
    if (tid < nNeur) atomic_fetch_sub_explicit(&lastF[tid], base, memory_order_relaxed);
    if (tid == 0) atomic_store_explicit(clk, 0u, memory_order_relaxed);
}
