#pragma once
// brain.h  –  core event-driven ABNN with fixed input/output blocks
// -----------------------------------------------------------------
// Layout: [ 0 … N_INPUT-1 ]  input neurons
//         [ N_INPUT … N_INPUT+N_OUTPUT-1 ]  output neurons
//         [ remaining ]  hidden neurons
// -----------------------------------------------------------------

#include <Metal/Metal.hpp>
#include <cstdint>
#include <vector>
#include <memory>

struct SynapsePacked   // matches Metal struct
{
    uint32_t src;
    uint32_t dst;
    float    w;
    float    pad;   // alignment
};

class Brain
{
public:
    // ---- compile-time constants ----------------------------------------
    static constexpr uint32_t kTickNS          = 10;          // 10-ns virtual tick
    static constexpr uint32_t kRenormThreshold = 0xE0000000u; // renorm near wrap

    // ---- construction ---------------------------------------------------
    // nInput + nOutput + nHidden = total neurons
    Brain(uint32_t nInput,
          uint32_t nOutput,
          uint32_t nHidden,
          uint32_t nSynapses,
          uint32_t eventsPerPass = 1'000'000);

    ~Brain();

    // ---- GPU bootstrapping ---------------------------------------------
    void build_pipeline(MTL::Device* device, MTL::Library* lib);
    void build_buffers (MTL::Device* device);

    // ---- Main GPU pass --------------------------------------------------
    void encode_traversal(MTL::CommandBuffer* cmdBuf);

    // ---- Host-side stimulation helpers ---------------------------------
    // analogue ∈ [0,1] length == N_INPUT; poissonHz caps firing rate
    void inject_inputs(const std::vector<float>& analogue, float poissonHz);

    // call after encode_traversal() has finished to know which outputs fired
    // returns vector<bool> size N_OUTPUT (true = at least one spike this pass)
    std::vector<bool> read_outputs() const;

    // ---- I/O ------------------------------------------------------------
    void save(std::ostream& os) const;
    void load(std::istream& is);

    // ---- dimensions -----------------------------------------------------
    uint32_t n_input()  const { return N_INPUT_;  }
    uint32_t n_output() const { return N_OUTPUT_; }
    uint32_t n_neuron() const { return N_NRN_;    }
    uint32_t n_syn()    const { return N_SYN_;    }

    // ---- raw Metal buffers (advanced) -----------------------------------
    MTL::Buffer* synapse_buffer()      const { return bufSyn_;   }
    MTL::Buffer* clock_buffer()        const { return bufClock_; }
    MTL::Buffer* last_fire_buffer()    const { return bufLastFire_; }
    MTL::Buffer* last_visit_buffer()   const { return bufLastVisit_; }

private:
    // immutable sizes
    uint32_t N_INPUT_, N_OUTPUT_, N_HIDDEN_, N_NRN_, N_SYN_, EVENTS_;

    // ---- host mirror of synapses (for save/load) ------------------------
    std::vector<SynapsePacked> hostSyn_;

    // ---- Metal buffers --------------------------------------------------
    MTL::Buffer* bufSyn_        { nullptr };
    MTL::Buffer* bufLastFire_   { nullptr };  // uint32 per neuron
    MTL::Buffer* bufLastVisit_  { nullptr };
    MTL::Buffer* bufClock_      { nullptr };  // uint32
    MTL::Buffer* bufRng_        { nullptr };

    // ---- pipeline states ------------------------------------------------
    MTL::ComputePipelineState* pipeTraverse_ { nullptr };
    MTL::ComputePipelineState* pipeRenorm_   { nullptr };

    // ---- helpers --------------------------------------------------------
    void release_all();
    void renormalise_if_needed(MTL::CommandBuffer* cb);

    // non-copyable
    Brain(const Brain&)            = delete;
    Brain& operator=(const Brain&) = delete;
};
