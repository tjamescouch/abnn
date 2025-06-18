#pragma once
/* brain.h  –  Host-side ABNN representation
 * ====================================================================
 * * Owns all Metal buffers and pipelines
 * * encode_traversal() enqueues one Monte-Carlo pass
 * * Exposes reward_buffer() and last_fired_buffer() for
 *   teacher-forcing / reward-modulated STDP.
 */

#include <Metal/Metal.hpp>
#include <vector>
#include <cstdint>
#include <istream>
#include <ostream>

/* -------- constants shared with kernel -------------------------------- */
static constexpr uint32_t kTickNS       = 1000;
static constexpr uint32_t kMaxSpikes    = 256;        /* exploration budget */
static constexpr uint32_t kRenormThresh = 4'000'000; /* renorm every 4 M   */

struct SynapsePacked { uint32_t src, dst; float w, pad; };

/* ===================================================================== */
class Brain
{
public:
    Brain(uint32_t nInput,
          uint32_t nOutput,
          uint32_t nHidden,
          uint32_t nSynapses,
          uint32_t eventsPerPass);
    ~Brain();

    /* one-time initialisation */
    void build_pipeline(MTL::Device*, MTL::Library*);
    void build_buffers (MTL::Device*);

    /* per-pass operations */
    void encode_traversal(MTL::CommandBuffer*);
    void inject_inputs(const std::vector<float>& vals, float hz);
    std::vector<bool> read_outputs() const;

    /* persistence */
    void save(std::ostream&) const;
    void load(std::istream&);

    /* getters ----------------------------------------------------------- */
    uint32_t n_input () const { return N_INPUT_;  }
    uint32_t n_output() const { return N_OUTPUT_; }
    uint32_t n_hidden() const { return N_HIDDEN_; }
    uint32_t n_neuron() const { return N_NRN_;    }
    uint32_t n_syn   () const { return N_SYN_;    }

    MTL::Buffer* synapse_buffer()     const { return bufSyn_;       }
    MTL::Buffer* last_fired_buffer()  const { return bufLastFire_;  }
    MTL::Buffer* clock_buffer()       const { return bufClock_;     }
    MTL::Buffer* reward_buffer()      const { return bufReward_;    }
    MTL::Buffer* budget_buffer()      const { return bufBudget_;    }

private:
    void release_all();
    void renormalise_if_needed(MTL::CommandBuffer*);

    /* immutable sizes */
    const uint32_t N_INPUT_, N_OUTPUT_, N_HIDDEN_;
    const uint32_t N_NRN_,   N_SYN_,    EVENTS_;

    /* Metal buffers */
    MTL::Buffer *bufSyn_       {nullptr};
    MTL::Buffer *bufLastFire_  {nullptr};
    MTL::Buffer *bufLastVisit_ {nullptr};
    MTL::Buffer *bufClock_     {nullptr};
    MTL::Buffer *bufBudget_    {nullptr};
    MTL::Buffer *bufReward_    {nullptr};
    MTL::Buffer *bufRBar_      {nullptr};

    /* pipelines */
    MTL::ComputePipelineState *pipeTrav_{nullptr};
    MTL::ComputePipelineState *pipeRenorm_{nullptr};

    /* host copy for inspection/debug */
    std::vector<SynapsePacked> hostSyn_;
};
