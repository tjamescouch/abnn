// brain.metal – ABNN kernels (32-bit ticks + renormalisation)
// ===========================================================
// Tick unit: 10 ns   (2³¹ ticks ≈ 21.47 s)
// When the host sees nowTick > 0xE0000000, it dispatches
// `renormalise_clock_and_times`, which subtracts that base
// from every timestamp and zeroes the global clock.
//
// 1) monte_carlo_traversal
// 2) renormalise_clock_and_times
// -----------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

struct SynapsePacked
{
    uint  src;
    uint  dst;
    float w;
    float pad;
};

// ───────────────────────────── RNG
inline uint lcg(thread uint& s){  s = s*1664525u + 1013904223u; return s; }
inline float lcg01(thread uint& s){ return float(lcg(s)) * (1.0f/4294967296.0f); }

// ───────────────────────────── 1) MONTE-CARLO
kernel void monte_carlo_traversal(
    device SynapsePacked* syns           [[ buffer(0) ]],
    device atomic_uint*   lastFired      [[ buffer(1) ]],
    device atomic_uint*   lastVisited    [[ buffer(2) ]],
    device atomic_uint&   nowTick        [[ buffer(3) ]],
    constant uint&        N_SYN          [[ buffer(4) ]],
    constant uint&        tau_pre_post   [[ buffer(5) ]], // ticks
    constant uint&        tau_visit      [[ buffer(6) ]], // ticks
    constant float&       alpha_LTP      [[ buffer(7) ]],
    constant float&       alpha_LTD      [[ buffer(8) ]],
    constant float&       w_min          [[ buffer(9) ]],
    constant float&       w_max          [[ buffer(10)]],
    device uint*          rngStates      [[ buffer(11)]],
    uint                  gid            [[ thread_position_in_grid ]]
){
    thread uint rng = rngStates[gid];
    uint edgeIdx    = lcg(rng) % N_SYN;
    SynapsePacked s = syns[edgeIdx];

    uint tick = atomic_fetch_add_explicit(&nowTick, 1u, memory_order_relaxed);

    uint dtSpike = tick - atomic_load_explicit(&lastFired  [s.src], memory_order_relaxed);
    uint dtVisit = tick - atomic_load_explicit(&lastVisited[s.dst], memory_order_relaxed);

    bool fire = (dtSpike < tau_pre_post) &&
                (s.w * exp(-float(dtVisit)/float(tau_visit)) > lcg01(rng));

    if (fire) {
        float newW = (dtSpike < tau_pre_post)
                   ? s.w + alpha_LTP * (1.0f - s.w)
                   : s.w - alpha_LTD * s.w;
        syns[edgeIdx].w = clamp(newW, w_min, w_max);
        atomic_store_explicit(&lastFired[s.dst], tick, memory_order_relaxed);
    }
    atomic_store_explicit(&lastVisited[s.dst], tick, memory_order_relaxed);
    rngStates[gid] = rng;
}

// Host will trigger this when it sees nowTick > 0xE0000000u
constant uint RENORM_THRESHOLD = 0xE0000000u;

// ───────────────────────────── 2) RENORMALISATION
kernel void renormalise_clock_and_times(
    device atomic_uint*   lastFired     [[ buffer(0) ]],
    device atomic_uint*   lastVisited   [[ buffer(1) ]],
    device atomic_uint&   nowTick       [[ buffer(2) ]],
    constant uint&        N_NRN         [[ buffer(3) ]],
    uint                  gid           [[ thread_position_in_grid ]],
    uint                  tid           [[ thread_index_in_threadgroup ]]
){
    // One uint in threadgroup memory to broadcast the base value
    threadgroup uint baseTG[1];

    if (tid == 0)
        baseTG[0] = atomic_exchange_explicit(&nowTick, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base = baseTG[0];
    if (base == 0u) return;          // nothing to do
    if (gid >= N_NRN) return;        // out-of-range threads (grid ≥ neurons)

    uint oldF = atomic_load_explicit(&lastFired  [gid], memory_order_relaxed);
    uint oldV = atomic_load_explicit(&lastVisited[gid], memory_order_relaxed);

    atomic_store_explicit(&lastFired  [gid], oldF - base, memory_order_relaxed);
    atomic_store_explicit(&lastVisited[gid], oldV - base, memory_order_relaxed);
}
