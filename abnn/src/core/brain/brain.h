#pragma once
/* brain.h  –  lightweight biologically-inspired event-driven NN core
   =================================================================
   Drop this header next to brain.cpp / brain.metal. It exposes the public
   interface used by BrainEngine and hides all Metal details inside.        */

#include <vector>
#include <cstdint>
#include <iosfwd>

namespace MTL {
class Device;
class Library;
class CommandBuffer;
class Buffer;
class ComputePipelineState;
}

/* compact synapse in GPU buffer ------------------------------------------------ */
struct SynapsePacked
{
    uint32_t src;
    uint32_t dst;
    float    w;
    float    pad;
};

/* ───────────────────────────────────────────────────────────────────────────── */
class Brain
{
public:
    static constexpr uint32_t kTickNS         = 1000;        /* 1 µs per tick   */
    static constexpr uint32_t kRenormThreshold= 1'000'000;   /* ticks           */
    static constexpr uint32_t kMaxSpikes      = 16;          /* spike budget    */

    /* construction ---------------------------------------------------------- */
    Brain(uint32_t nInput,  uint32_t nOutput,
          uint32_t nHidden, uint32_t nSynapses,
          uint32_t eventsPerKernel);
    ~Brain();

    /* one-time GPU initialisation ------------------------------------------ */
    void build_pipeline(MTL::Device* dev, MTL::Library* lib);
    void build_buffers (MTL::Device* dev);

    /* per-pass GPU encoding ------------------------------------------------- */
    void encode_traversal(MTL::CommandBuffer* cb);

    /* drive & read ---------------------------------------------------------- */
    void inject_inputs(const std::vector<float>& analogue, float poissonHz);
    std::vector<bool> read_outputs() const;

    /* persistence ----------------------------------------------------------- */
    void save(std::ostream& os) const;
    void load(std::istream& is);

    /* getters --------------------------------------------------------------- */
    uint32_t n_input () const { return N_INPUT_;  }
    uint32_t n_output() const { return N_OUTPUT_; }
    uint32_t n_hidden() const { return N_HIDDEN_; }
    uint32_t n_neuron() const { return N_NRN_;    }
    uint32_t n_syn   () const { return N_SYN_;    }

    MTL::Buffer* synapse_buffer() const { return bufSyn_; }
    MTL::Buffer* budget_buffer() const { return bufBudget_; }

private:
    void release_all();
    void renormalise_if_needed(MTL::CommandBuffer* cb);

    /* network sizes --------------------------------------------------------- */
    const uint32_t N_INPUT_, N_OUTPUT_, N_HIDDEN_;
    const uint32_t N_NRN_,   N_SYN_;
    const uint32_t EVENTS_;

    /* Metal resources ------------------------------------------------------- */
    MTL::Buffer* bufSyn_       {nullptr};   /* synapse array (managed)   */
    MTL::Buffer* bufLastFire_  {nullptr};   /* per-neuron last spike ts  */
    MTL::Buffer* bufLastVisit_ {nullptr};   /* (unused in current kern)  */
    MTL::Buffer* bufClock_     {nullptr};   /* global tick counter       */
    MTL::Buffer* bufBudget_    {nullptr};   /* remaining spikes budget   */

    MTL::ComputePipelineState* pipeTraverse_{nullptr};
    MTL::ComputePipelineState* pipeRenorm_  {nullptr};

    /* CPU-side copy of synapses for save/load convenience ------------------ */
    std::vector<SynapsePacked> hostSyn_;
};
