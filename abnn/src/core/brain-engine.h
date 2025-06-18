#pragma once
/* brain-engine.h  –  harness driving a Brain instance
 * =====================================================================
 * Implements
 *   • teacher forcing
 *   • sliding-window loss  (window = 1000 passes ≈ 1 s)
 *   • graded reward  (loss decrease → positive reward)
 *
 * Public API
 *   BrainEngine(device,nInput,nOutput [,eventsPerPass])
 *   set_stimulus(shared_ptr<StimulusProvider>)
 *   start_async() / stop_async()
 */

#include <Metal/Metal.hpp>
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <cstdint>
#include "rate-filter.h"
#include "constants.h"




/* forward decls -------------------------------------------------------- */
class Brain;
class Logger;
class StimulusProvider;

/* ===================================================================== */
class BrainEngine
{
public:
    BrainEngine(MTL::Device* device,
                uint32_t     nInput,
                uint32_t     nOutput,
                uint32_t     eventsPerPass = EVENTS_PER_PASS);
    ~BrainEngine();

    /* attach stimulus generator BEFORE start_async() */
    void set_stimulus(std::shared_ptr<StimulusProvider> stim);

    /* non-blocking background loop */
    void start_async();
    void stop_async();
    
    /* model persistence (binary .bnn) */
    bool  load_model(const std::string& filename = "");
    bool  save_model(const std::string& filename = "") const;

private:
    float maxObserved = 0.5f;     // initialize to expected plateau

    /* single synchronous simulation pass */
    std::vector<bool> run_one_pass();

    /* Metal handles ---------------------------------------------------- */
    MTL::Device*       device_{nullptr};
    MTL::CommandQueue* commandQueue_{nullptr};
    MTL::Library*      defaultLib_{nullptr};

    /* ABNN + logging --------------------------------------------------- */
    std::unique_ptr<Brain>  brain_;
    std::unique_ptr<Logger> logger_;
    std::shared_ptr<StimulusProvider> stim_;

    /* async thread ----------------------------------------------------- */
    std::thread       worker_;
    std::atomic<bool> running_{false};

    /* dimensions / params --------------------------------------------- */
    uint32_t nIn_{0};
    uint32_t nOut_{0};
    uint32_t eventsPerPass_{0};

    /* sliding-window loss state --------------------------------------- */
    std::vector<uint32_t> spikeWindow_;   /* counts per output neuron   */
    size_t winPos_{0};
    const size_t WIN_SIZE_ = 1000;        /* 1000 passes ≈ 1 s          */

    double lastLoss_{0.25};               /* baseline for graded reward */
    RateFilter rateFilter_{ /*τ=*/FILTER_TAU, /*useFIR=*/USE_FIR };
};
