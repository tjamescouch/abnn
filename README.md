1 Overview
Aspect	Goal
Computation style	Event-driven Monte-Carlo traversal rather than synchronous “layered” ticks
Topology	Sparse directed graph
Nodes ≈ neurons, edges ≈ chemical synapses
State variables	Per-synapse weight, per-neuron last-fired timestamp, per-neuron last-visited timestamp
Plasticity	Spike-Timing-Dependent Plasticity (STDP) on each traversal step
Execution target	① CPU prototype (C++ 17) ② Metal-compute pipeline (MPS or Metal CPP)

2 Core Data Model
All arrays are flat, contiguous, 64-bit aligned so they can be copied directly into GPU buffers.

2.1 Header Section
Offset	Type	Name	Description
0	uint32_t	N_SYN	Total synapses
4	uint32_t	N_NRN	Total neurons
8 → 15	padding	—	Keep next array 16-byte aligned

Binary layout tip – Treat the header as a 16-byte struct; everything that follows can be reinterpret_cast on GPU.

2.2 Array Section (all std::vector<T> on CPU)
Array	Type	Size	Semantics
synapses	uint32_t[2]	N_SYN × 2	src, dst neuron indices
weights	float32	N_SYN	Synaptic efficacy
lastFiredNS	uint64_t	N_NRN	Last spike time (ns since start)
lastVisitedNS	uint64_t	N_NRN	Last Monte-Carlo visit time

GPU note – Combine synapses + weights into a SoA to enable coalesced reads:
struct SynapsePacked { uint32_t src; uint32_t dst; float32 w; float32 pad; };

3 Global Clock
Monotonic nanosecond counter incremented per simulated event, not wall-time.

text
Copy
uint64_t NOW_NS = 0;
const uint64_t ΔT_EVENT = 1;   // ns advanced per accepted traversal step
• Keeps arithmetic on integers → deterministic, rollback-friendly.
• Enables virtual-time Monte-Carlo so thousands of events can occur in a real-time millisecond.

4 Monte-Carlo Traversal Algorithm
pseudocode
Copy
for step in 0 .. STEPS-1:
    edgeIdx = uniformInt(0, N_SYN-1)      // ① pick random synapse
    src = synapses[edgeIdx].src
    dst = synapses[edgeIdx].dst

    shouldFire = evaluateFire(src, dst)    // ② causal test
    if shouldFire:
        fire(src, dst, edgeIdx)            // ③ apply spike + plasticity

    lastVisitedNS[dst] = NOW_NS            // ④ update visitation
    NOW_NS += ΔT_EVENT                     // ⑤ advance global time
4.1 evaluateFire
pseudocode
Copy
Δt_visit   = NOW_NS - lastVisitedNS[dst]
Δt_spike   = NOW_NS - lastFiredNS[src]
threshold  = baseThresh * exp(-Δt_visit / τ_dendritic)

return Δt_spike < τ_pre_post and weights[edgeIdx] > randomFloat()
Intuition

Recency of visit approximates membrane potential decay.

Δt_spike < τ_pre_post captures causal pairing needed for STDP.

Weight comparison adds stochasticity resembling vesicle release probability.

4.2 fire
pseudocode
Copy
lastFiredNS[dst] = NOW_NS

// STDP update
if NOW_NS - lastFiredNS[src] < τ_LTP:
    weights[edgeIdx] += α_LTP * (1 - weights[edgeIdx])
else:
    weights[edgeIdx] -= α_LTD * weights[edgeIdx]

// Homeostatic clipping
weights[edgeIdx] = clamp(weights[edgeIdx], w_min, w_max)
5 Plasticity & Rewiring
Mechanism	Trigger	Action
STDP	Pairing within τ_LTP / τ_LTD windows	Potentiate / depress current weight
Synapse pruning	weights[i] < w_prune	Remove entry; compact array periodically
Synaptogenesis	rand() < p_new on fire	Append new synapse (src,dst') with w_init
Neuron addition	Off-line growth phase	Reallocate arrays and adjust header

6 Initialization
Allocate arrays.

Populate synapses via Erdős–Rényi or small-world generator.

Draw initial weights from Beta(2,8) (skewed low).

Set lastFiredNS[:] = lastVisitedNS[:] = 0.

Provide a YAML manifest to reproduce runs:

yaml
Copy
neurons:  65536
synapses: 524288
tau_LTP:  20_000       # ns
tau_LTD:  40_000
alpha_LTP: 0.01
alpha_LTD: 0.005
w_min: 0.001
w_max: 1.0
steps: 1_000_000
rng_seed: 42
7 Parallel Execution Strategy
7.1 CPU Prototype
Sharded Monte-Carlo on std::thread pools: each worker owns a slice of synapses.
Use lock-free atomic fetch-add for NOW_NS.

7.2 Metal / MPS
Buffer	Access	Threadgroup size
SynapsePacked	read-only	256
NeuronTimes	read-write	256
RNG State	private per-thread	1

Kernel sketch:

metal
Copy
kernel void monteCarloTraversal(
    device SynapsePacked* syns     [[buffer(0)]],
    device uint64_t*      lastF    [[buffer(1)]],
    device uint64_t*      lastV    [[buffer(2)]],
    device atomic_uint64_t& nowNS  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
){
    uint edgeIdx = pcg32(&rngState) % N_SYN;
    SynapsePacked s = syns[edgeIdx];
    ...
}
• Atomic increments on a single 64-bit nowNS avoid race conditions.
• For very large graphs, launch many tiny kernels instead of one monolith—avoids long GPU fences.

8 Pseudocode API (C++-17)
cpp
Copy
struct GraphBNN {
    uint32_t N_SYN, N_NRN;
    std::vector<uint32_t> src, dst;
    std::vector<float>    w;
    std::vector<uint64_t> lastFired, lastVisited;
    uint64_t nowNS = 0;

    void step(size_t iterations);
    void fire(uint32_t edgeIdx, uint32_t src, uint32_t dst);
    bool shouldFire(uint32_t edgeIdx, uint32_t src, uint32_t dst);
};
step() wraps the loop from §4; expose it to Python via pybind11 for quick experiments.

9 I/O & Serialization
File	Format	Notes
.bnn	Raw binary as per §2	Small header + 4 arrays
.yaml	Human-editable hyper-parameters	Optional; embed SHA-256 of YAML in .bnn footer for provenance

10 Testing & Validation
Unit – Deterministic RNG seeds yield identical weight trajectories.

Statistical – Weight distribution over time converges to log-normal.

Biological – Pairwise spike correlation obeys experimentally observed 10 ms window (see Bi & Poo 1998).

Performance – >10 M synaptic events / s on Apple M3 Ultra (Metal).

11 Extensibility Road-Map
Stage	Feature	Rationale
v0.2	Multiple neurotransmitter types (excitatory / inhibitory)	Balance network activity
v0.3	Short-term facilitation / depression	Model vesicle depletion
v0.4	Neuromodulators (dopamine)	Reward-modulated STDP
v0.5	Structural plasticity on GPU	Real-time growth / pruning

12 Glossary
Symbol	Meaning
τ_LTP, τ_LTD	Time constants for Long-Term Potentiation / Depression
α_LTP, α_LTD	Learning rates
NOW_NS	Global virtual nanosecond clock
ΔT_EVENT	Clock increment per Monte-Carlo step


