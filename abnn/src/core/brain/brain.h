#pragma once
// brain.h  –  ABNN core (CPU ⇄ GPU bridge)

#include <Metal/Metal.hpp>
#include <cstdint>
#include <vector>
#include <memory>
#include <ostream>
#include <istream>

/*───────────────────────────────────────────────────────────────────────────────
 * Flat synapse packing on both CPU & GPU.
 *─────────────────────────────────────────────────────────────────────────────*/
struct SynapsePacked
{
    uint32_t src;
    uint32_t dst;
    float    w;
    float    pad;   // alignment
};

class Brain
{
public:
    /*  10-ns tick → 2³¹ ticks ≈ 21.47 s before wrap  */
    static constexpr uint32_t kTickNS           = 10;
    static constexpr uint32_t kRenormThreshold  = 0xE0000000u;   // ~94 % of 2³²

    Brain(uint32_t neurons,
          uint32_t synapses,
          uint32_t eventsPerPass = 1'000'000);
    ~Brain();

    void buildPipeline(MTL::Device* device, MTL::Library* library);
    void buildBuffers (MTL::Device* device);

    void step (MTL::CommandBuffer* cmdBuf);            // GPU advance
    void stepCPU(size_t iterations = 1);               // debugging

    void save(std::ostream& os) const;
    void load(std::istream& is);

    uint32_t neuronCount()   const { return N_NRN_; }
    uint32_t synapseCount()  const { return N_SYN_; }
    uint32_t eventsPerPass() const { return eventsPerPass_; }

    MTL::Buffer* synapseBuffer()       const { return bufferSynapses_;     }
    MTL::Buffer* lastFiredBuffer()     const { return bufferLastFired_;    }
    MTL::Buffer* lastVisitedBuffer()   const { return bufferLastVisited_;  }
    MTL::Buffer* clockBuffer()         const { return bufferClock_;        }
    MTL::Buffer* rngStateBuffer()      const { return bufferRngStates_;    }

private:
    uint32_t N_NRN_;
    uint32_t N_SYN_;
    uint32_t eventsPerPass_;

    std::vector<SynapsePacked> hostSynapses_;

    /* Metal buffers */
    MTL::Buffer* bufferSynapses_    { nullptr };
    MTL::Buffer* bufferLastFired_   { nullptr };
    MTL::Buffer* bufferLastVisited_ { nullptr };
    MTL::Buffer* bufferClock_       { nullptr };
    MTL::Buffer* bufferRngStates_   { nullptr };

    /* Pipeline states */
    MTL::ComputePipelineState* mcPipeline_    { nullptr };  // traversal
    MTL::ComputePipelineState* renormPipeline_{ nullptr };  // NEW ★

    /* Non-copyable */
    Brain(const Brain&)            = delete;
    Brain& operator=(const Brain&) = delete;
};
