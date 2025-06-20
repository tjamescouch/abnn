# ABNN: Reward-Modulated Spiking Neural Network

A biologically inspired, stochastic spiking neural network implemented in modern C++ (C++17) and Metal. ABNN uses event-driven Monte Carlo traversal, STDP, homeostatic plasticity, and reward modulation to learn arbitrary functional mappings.

---

## 1. Overview

| Aspect              | Goal                                                                                   |
| ------------------- | -------------------------------------------------------------------------------------- |
| **Computation**     | Event-driven Monte Carlo traversal rather than synchronous “layered” ticks             |
| **Topology**        | Sparse directed graph (neurons ⇄ nodes, synapses ⇄ edges)                              |
| **State variables** | Per-synapse weight; per-neuron last-fired timestamp; per-neuron last-visited timestamp |
| **Plasticity**      | Spike-Timing-Dependent Plasticity (STDP) on each traversal step                        |
| **Targets**         | Metal compute (Metal-CPP / MPS)                           |

---

## 2. Core Data Model

All arrays are flat, contiguous, 64-bit aligned so they can be mapped directly into GPU buffers.

### 2.1 Header Section

```
Offset | Type      | Name   | Description
-------|-----------|--------|----------------------------------------
0      | uint32_t  | N_SYN  | Total number of synapses               
4      | uint32_t  | N_NRN  | Total number of neurons                
8–15   | padding   | —      | Aligns next array on 16-byte boundary  
```

> **Tip:** Treat the header as a 16-byte struct; everything that follows can be reinterpret\_cast in GPU shaders.

### 2.2 Array Section (CPU: `std::vector<T>`)

| Array           | Type                   | Size     | Semantics                                   |
| --------------- | ---------------------- | -------- | ------------------------------------------- |
| `synapses`      | `(uint32_t, uint32_t)` | N\_SYN×2 | Source and destination neuron indices       |
| `weights`       | `float32`              | N\_SYN   | Synaptic efficacies                         |
| `lastFiredNS`   | `uint64_t`             | N\_NRN   | Timestamp of last spike per neuron (ns)     |
| `lastVisitedNS` | `uint64_t`             | N\_NRN   | Last Monte Carlo visit timestamp per neuron |

> **GPU layout:** Combine synapses and weights for coalesced reads:
>
> ```cpp
> struct SynapsePacked {
>   uint32_t src;
>   uint32_t dst;
>   float    w;
>   float    pad;
> };
> ```

---

## 3. Global Clock

A monotonic nanosecond counter incremented per simulated event (not real wall-time):

```cpp
uint64_t NOW_NS = 0;
const uint64_t ΔT_EVENT = 1;  // ns advanced per accepted traversal step
```

* Integer arithmetic ensures determinism and easy rollbacks.
* Virtual-time Monte Carlo lets thousands of events simulate within a real-time millisecond.

---

## 4. Monte Carlo Traversal Algorithm

**High-level pseudocode:**

```pseudo
for step in 0 .. STEPS-1:
    edgeIdx = uniformInt(0, N_SYN-1)      // pick a random synapse
    src = synapses[edgeIdx].src
    dst = synapses[edgeIdx].dst

    if evaluateFire(src, dst, edgeIdx):   // causal test
        fire(src, dst, edgeIdx)           // apply spike & plasticity

    lastVisitedNS[dst] = NOW_NS          // update visitation time
    NOW_NS += ΔT_EVENT                   // advance clock
```

### 4.1 `evaluateFire`

```pseudo
Δt_visit  = NOW_NS - lastVisitedNS[dst]
Δt_spike  = NOW_NS - lastFiredNS[src]
threshold = baseThresh * exp(-Δt_visit / τ_dendritic)

// Causal & stochastic firing decision
return (Δt_spike < τ_pre_post) && (weights[edgeIdx] > randomFloat())
```

* **Membrane decay:** recency of visit approximates dendritic potential.
* **STDP window:** Δt\_spike < τ\_pre\_post enforces causality.
* **Stochasticity:** weight comparison simulates vesicle release probability.

### 4.2 `fire`

```pseudo
lastFiredNS[dst] = NOW_NS

// STDP weight update
if (NOW_NS - lastFiredNS[src] < τ_LTP):
    weights[edgeIdx] += α_LTP * (1 - weights[edgeIdx])
else:
    weights[edgeIdx] -= α_LTD * weights[edgeIdx]

// Homeostatic clipping
theta = clamp(weights[edgeIdx], w_min, w_max)
```

---

## 5. Plasticity & Rewiring

| Mechanism           | Trigger                        | Action                                         |
| ------------------- | ------------------------------ | ---------------------------------------------- |
| **STDP**            | Spike pairing within τ windows | Potentiate or depress current weight           |
| **Synapse pruning** | `weights[i] < w_prune`         | Remove synapse; compact array periodically     |
| **Synaptogenesis**  | `rand() < p_new` on fire       | Append new synapse `(src, dst')` with `w_init` |
| **Neuron addition** | Off-line growth phase          | Reallocate arrays; adjust header               |

---

## 6. Initialization

1. Allocate arrays (`N_NRN`, `N_SYN`).
2. Generate connectivity (Erdős–Rényi or small-world).
3. Initialize `weights` \~ Beta(2,8) (skewed low).
4. Zero `lastFiredNS` and `lastVisitedNS`.
5. constants.h defines hyperparameters:

```cpp

#define NUM_INPUTS  256
#define NUM_OUTPUTS 256
#define NUM_HIDDEN 5'000'000
#define NUM_SYN    1'000'000'000

#define INPUT_SIN_WAVE_FREQUENCY 0.5

#define INPUT_RATE_HZ 1000
#define PEAK_DECAY 0.999f         // how quickly old peaks fade
#define EVENTS_PER_PASS 150'000'000
#define FILTER_TAU 0.02
#define USE_FIR true
#define dT_SEC 0.0009

#define _aLTP 0.04f
#define _aLTD 0.02f
#define _wMin 0.001f
#define _wMax 1.0f

```

---

## 7. Parallel Execution Strategy

### 7.1 CPU Prototype

* Sharded Monte Carlo loops on `std::thread` pools.
* Lock-free atomic fetch-add for `NOW_NS`.

### 7.2 Metal / MPS

| Buffer          | Access     | Threadgroup size |
| --------------- | ---------- | ---------------- |
| `SynapsePacked` | read-only  | 256              |
| `NeuronTimes`   | read-write | 256              |
| `rngState`      | private    | 1                |

**Kernel sketch:**

```metal
kernel void monteCarloTraversal(
    device SynapsePacked*    syns      [[buffer(0)]],
    device uint64_t*         lastF     [[buffer(1)]],
    device uint64_t*         lastV     [[buffer(2)]],
    device atomic_uint64_t&  nowNS     [[buffer(3)]],
    uint                     gid       [[thread_position_in_grid]]
){
    uint edgeIdx = pcg32(&rngState) % N_SYN;
    auto s = syns[edgeIdx];
    ...
    atomic_fetch_add_explicit(&nowNS, ΔT_EVENT, memory_order_relaxed);
}
```

* Atomic increments avoid races on the global clock.
* For huge graphs, launch many smaller kernels to reduce GPU fence times.

---

## 8. Pseudocode API (C++17)

```cpp
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
```

* `step()` runs the Monte Carlo loop.
* Expose via `pybind11` for rapid prototyping.

---

## 9. I/O & Serialization

| File    | Format | Notes                                                   |
| ------- | ------ | ------------------------------------------------------- |
| `.bnn`  | Binary | Header + flat arrays (see §2)                           |

---

## 10. Testing & Validation

* **Unit tests:** Fixed RNG seeds produce identical weight trajectories.
* **Statistical:** Weight distributions converge to log-normal.
* **Biological:** Pairwise spike correlations match 10 ms STDP windows (Bi & Poo 1998).
* **Performance:** >10M synaptic events/sec on Apple M3 Ultra (Metal).

Running the project will currently learn sine→cos² mapping for an input and target signal that phase shifts temporally:
<img width="556" alt="image" src="https://github.com/user-attachments/assets/70e74a1d-74e1-44cc-b65b-efa28ec8f82d" />

---

## 11. Extensibility Roadmap

| Version | Feature                            | Rationale                        |
| ------- | ---------------------------------- | -------------------------------- |
| v0.2    | Excitatory/Inhibitory neurons      | Balance network dynamics         |
| v0.3    | Short-term facilitation/depression | Model vesicle depletion          |
| v0.4    | Dopaminergic neuromodulation       | Reward-modulated STDP            |
| v0.5    | Structural plasticity on GPU       | Real-time synapse growth/pruning |

---

## 12. Glossary

| Symbol         | Meaning                                                     |
| -------------- | ----------------------------------------------------------- |
| τ\_LTP, τ\_LTD | Time constants for Long-Term Potentiation / Depression (ns) |
| α\_LTP, α\_LTD | STDP learning rates                                         |
| NOW\_NS        | Global virtual-time nanosecond clock                        |
| ΔT\_EVENT      | Clock increment per Monte-Carlo step (ns)                   |

---

