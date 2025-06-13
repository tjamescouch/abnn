#include <metal_stdlib>
using namespace metal;

struct SynapsePacked {
    uint  src;
    uint  dst;
    float w;
    float pad;
};

// ---------------------------------------------------------------------------
// Tunables (start values give ~10-30 % of outputs spiking)
// ---------------------------------------------------------------------------
#define BASE_SCALE     0.15f    // baseline probability = w × BASE_SCALE
#define NOVELTY_COEF   0.25f    // extra boost for long-unvisited dst
#define REFRACTORY     2u       // ticks a neuron must stay silent after spike
#define MIN_WEIGHT     0.02f    // floor for probability calculation
#define CLOCK_INC      1u       // ticks added per kernel pass
// ---------------------------------------------------------------------------

inline float fast_rand(uint state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (float)(state & 0x00FFFFFFu) / 16777216.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1 : Monte-Carlo traversal
// ─────────────────────────────────────────────────────────────────────────────
kernel void monte_carlo_traversal(
    device const SynapsePacked* synapses      [[ buffer(0) ]],
    device       atomic_uint*   lastFired     [[ buffer(1) ]],
    device       atomic_uint*   lastVisited   [[ buffer(2) ]],
    device       atomic_uint*   clockPtr      [[ buffer(3) ]],
    constant     uint&          nSynapses     [[ buffer(4) ]],
    constant     uint&          tau_vis       [[ buffer(5) ]],
    constant     uint&          tau_pre       [[ buffer(6) ]],
    constant     float&         aLTP          [[ buffer(7) ]],
    constant     float&         aLTD          [[ buffer(8) ]],
    constant     float&         wMin          [[ buffer(9) ]],
    constant     float&         wMax          [[ buffer(10)]],
    device       uint*          rngBuf        [[ buffer(11)]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= nSynapses) return;

    SynapsePacked s = synapses[tid];
    uint now = atomic_load_explicit(clockPtr, memory_order_relaxed);

    // ---------------- Refractory: skip if dst fired very recently ----------
    uint lastDst = atomic_load_explicit(&lastFired[s.dst], memory_order_relaxed);
    if (now - lastDst <= REFRACTORY) return;

    // ---------------- Base + novelty probability ---------------------------
    float p = max(s.w, MIN_WEIGHT) * BASE_SCALE;

    uint dtVis = now - atomic_load_explicit(&lastVisited[s.dst],
                                            memory_order_relaxed);
    float novelty = 1.0f - exp(- (float)dtVis / (float)tau_vis);
    p += s.w * NOVELTY_COEF * novelty;
    p = clamp(p, 0.0f, 1.0f);

    // ---------------- Bernoulli trial --------------------------------------
    float r = fast_rand(tid ^ now);
    if (p > r)
    {
        atomic_store_explicit(&lastFired[s.dst],   now, memory_order_relaxed);
        atomic_store_explicit(&lastVisited[s.dst], now, memory_order_relaxed);

        // Simple Hebbian LTP
        float newW = clamp(s.w + aLTP * (1.f - s.w), wMin, wMax);
        ((device SynapsePacked*)synapses)[tid].w = newW;
    }

    // ---------------- Advance global clock once per kernel pass ------------
    if (tid == 0)
        atomic_fetch_add_explicit(clockPtr, CLOCK_INC, memory_order_relaxed);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2 : Renormalise clock + per-neuron times
// ─────────────────────────────────────────────────────────────────────────────
kernel void renormalise_clock_and_times(
    device atomic_uint* lastFired   [[ buffer(0) ]],
    device atomic_uint* lastVisited [[ buffer(1) ]],
    device atomic_uint* clockPtr    [[ buffer(2) ]],
    constant uint&      nNeurons    [[ buffer(3) ]],
    uint tid [[thread_position_in_grid]]
)
{
    uint base = atomic_load_explicit(clockPtr, memory_order_relaxed);

    if (tid < nNeurons) {
        atomic_fetch_sub_explicit(&lastFired[tid],   base, memory_order_relaxed);
        atomic_fetch_sub_explicit(&lastVisited[tid], base, memory_order_relaxed);
    }
    if (tid == 0)
        atomic_store_explicit(clockPtr, 0u, memory_order_relaxed);
}
