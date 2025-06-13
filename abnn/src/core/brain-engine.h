#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>
#include <string>
#include <memory>

#include "brain.h"          // ← the class we just built

/*───────────────────────────────────────────────────────────────────────────────
 * BrainEngine
 *   • owns a Brain instance
 *   • handles Metal command-queue / library setup
 *   • steps the simulation for a requested number of events
 *   • triggers renormalisation when the clock nears wrap
 *   • can save / load the network state
 *─────────────────────────────────────────────────────────────────────────────*/
class BrainEngine
{
public:
    /* Construct an empty engine (no graph yet). */
    BrainEngine(MTL::Device*  device,
                uint32_t      eventsPerPass = 1'000'000);   // kernel batch size
    ~BrainEngine();

    /* Load a .bnn file into the Brain (creates buffers if needed). */
    bool loadModel(const std::string& path);

    /* Save current state to a .bnn file. */
    bool saveModel(const std::string& path) const;

    /* Step the simulation by (passes * eventsPerPass) Monte-Carlo events. */
    void run(uint32_t passes);

    /* Accessors */
    Brain& brain()               { return *brain_; }
    const Brain& brain() const   { return *brain_; }

private:
    /* Internal helpers */
    void buildMetalObjects();
    void ensureRenormalised(MTL::CommandBuffer* cmdBuf);

    /* Metal handles */
    MTL::Device*             device_          { nullptr };
    MTL::CommandQueue*       commandQueue_    { nullptr };
    MTL::Library*            computeLibrary_  { nullptr };
    MTL::ComputePipelineState* renormPipeline_{ nullptr };   // compiled once

    /* Core brain + buffers */
    std::unique_ptr<Brain>   brain_;
    bool                     buffersBuilt_ = false;

    /* Simulation parameters */
    uint32_t eventsPerPass_;   // kernel grid size per run() iteration
};
