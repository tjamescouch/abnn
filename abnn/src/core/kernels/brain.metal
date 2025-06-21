// brain.metal  –  Monte-Carlo traversal with
//                 • global spike-budget
//                 • Hebbian STDP  (+/-)
//                 • homeostatic weight drift toward a target firing rate
//                 • reward-modulated plasticity (three-factor rule)

#include <metal_stdlib>
using namespace metal;

/* packed synapse record (identical layout on CPU) */
struct SynapsePacked { uint src, dst; float w, pad; };
/* ------------------------------------------------------------------ *
   Debug counters shared between CPU and GPU
   Size = 3 × 4 bytes = 12; round up to 16 for alignment safety.
 * ------------------------------------------------------------------ */
struct DBG
{
    atomic_uint rewardHits;   // how many reward-phase passes we saw
    atomic_uint dwTeacher;    // ∑ |Δw| in teacher passes (scaled)
    atomic_uint dwReward;     // ∑ |Δw| in reward passes  (scaled)
    // padding not strictly required, but we can add one to make it 16 B
    atomic_uint _pad;         // keeps 16-byte alignment across compilers
};

enum PassType : uint
{
    PASS_TEACHER = 0,
    PASS_REWARD  = 1
};

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
#define ALPHA_RBAR    0.005f     /* EWMA for running reward average     */
    
#define NUM_INPUTS  256 //FIXME - pass these in they are currently duplicated
#define NUM_OUTPUTS 256



/* ===================================================================== *
   monte_carlo_traversal
   ---------------------------------------------------------------------
   • per-TG clock cache  (tgNow)
   • unique time-stamp per thread (now  = tgNow + tPos.x)
   • reward/debug counters
   • exploration boost that is:
         – linear in w
         – higher (×2) for synapses that end on output neurons
         – scaled by a runtime parameter   exploreScale  (buffer-16)
   • writes pre-synaptic last-fire ONLY when the synapse actually fired
 * ===================================================================== */

kernel void monte_carlo_traversal(
    device       SynapsePacked* syn   [[ buffer(0) ]],
    device atomic_uint*  lastF        [[ buffer(1) ]],
    device atomic_uint*  lastV        [[ buffer(2) ]],          /* unused */
    device atomic_uint*  clock        [[ buffer(3) ]],
    constant uint&       nSyn         [[ buffer(4) ]],
    constant uint&       tau_vis      [[ buffer(5) ]],
    constant uint&       tau_pre      [[ buffer(6) ]],
    constant float&      aLTP         [[ buffer(7) ]],
    constant float&      aLTD         [[ buffer(8) ]],
    constant float&      wMin         [[ buffer(9) ]],
    constant float&      wMax         [[ buffer(10)]],
    device atomic_uint*  budget       [[ buffer(11)]],
    device const float*  reward       [[ buffer(12)]],
    device atomic_float* rBarA        [[ buffer(13)]],
    device DBG*          dbg          [[ buffer(14)]],
    constant uint&       passType     [[ buffer(15)]],
    constant float&      exploreScale [[ buffer(16)]],   /*  ◀ new  */
    constant uint&       nInput       [[ buffer(17)]],   /*  ◀ new  */
    constant uint&       nOutput      [[ buffer(18)]],   /*  ◀ new  */
    uint3  tPos [[thread_position_in_threadgroup]],
    uint3  gPos [[thread_position_in_grid]],
    uint3  tgSz [[threads_per_threadgroup]])
{
    const uint tid = gPos.x;
    if (tid >= nSyn) return;

    /* ── shared clock ------------------------------------------------- */
    threadgroup uint tgNow = 0;
    if (tPos.x == 0)
        tgNow = atomic_load_explicit(clock, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint now = tgNow + tPos.x;                       /* unique per thread */

    /* ── debug: reward-hit counter ----------------------------------- */
    if (tPos.x == 0 && passType == PASS_REWARD)
        atomic_fetch_add_explicit(&dbg->rewardHits, 1u, memory_order_relaxed);

    SynapsePacked s = syn[tid];

    /* output-layer helper -------------------------------------------- */
    bool isOutSyn = (s.dst >= nInput) && (s.dst < nInput + nOutput);

#define ONE_TICK() \
    if (tid == 0)  \
        atomic_fetch_add_explicit(clock, CLOCK_INC, memory_order_relaxed)

    /* quick gates ---------------------------------------------------- */
    uint lp = atomic_load_explicit(&lastF[s.src], memory_order_relaxed);
    if (now - lp > WINDOW_PRE) { ONE_TICK(); return; }

    uint ld = atomic_load_explicit(&lastF[s.dst], memory_order_relaxed);
    if (now - ld <= REFRACTORY) { ONE_TICK(); return; }

    if (atomic_load_explicit(budget, memory_order_relaxed) == 0u) {
        ONE_TICK(); return;
    }

    /* spike probability --------------------------------------------- */
    float p;
    if (passType == PASS_REWARD) {
        float boost = isOutSyn ? (exploreScale * 2.0f) : exploreScale;
        p = clamp(s.w * boost, 0.f, 1.f);           // linear boost
    } else {
        p = clamp(s.w * s.w * BASE_SCALE, 0.f, 1.f);/* quadratic (teacher) */
    }

    bool fired = (p > rand01(tid ^ now));

    /* global budget race -------------------------------------------- */
    if (fired) {
        uint old = atomic_fetch_sub_explicit(budget, 1u, memory_order_relaxed);
        if (old == 0u) fired = false;                /* lost race   */
    }

    /* plasticity core ----------------------------------------------- */
    float dW = fired ? (aLTP * (1.f - s.w))
                     : (-aLTD * s.w);

    float R    = *reward;
    float rBar = atomic_load_explicit(rBarA, memory_order_relaxed);
    dW += ETA_REWARD * (R - rBar) * (fired ? 1.f : 0.f);

    if (tid == 0) {
        float newBar = rBar + ALPHA_RBAR * (R - rBar);
        atomic_store_explicit(rBarA, newBar, memory_order_relaxed);
    }

    /* homeostatic term ---------------------------------------------- */
    float isi   = float(now - ld);
    float estHz = isi > 0.f ? 1e6f / isi : 0.f;
    dW += ETA_HOME * (TARGET_RATE_HZ - estHz) * s.w;

    /* debug |dW| once per TG ---------------------------------------- */
    if (tPos.x == 0) {
        uint scaled = uint(fabs(dW) * 1e7f);
        if (passType == PASS_REWARD)
            atomic_fetch_add_explicit(&dbg->dwReward,  scaled, memory_order_relaxed);
        else
            atomic_fetch_add_explicit(&dbg->dwTeacher, scaled, memory_order_relaxed);
    }

    /* weight update -------------------------------------------------- */
    s.w = clamp(s.w + dW, wMin, wMax);
    syn[tid] = s;

    /* timestamps ----------------------------------------------------- */
    if (fired) {
        atomic_store_explicit(&lastF[s.src], now, memory_order_relaxed);
        atomic_store_explicit(&lastF[s.dst], now, memory_order_relaxed);
    }

    ONE_TICK();
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
