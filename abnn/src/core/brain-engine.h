#pragma once
// brain-engine.h  –  asynchronous harness driving Brain
// ======================================================

#include <Metal/Metal.hpp>
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <string>

#include "brain.h"
#include "logger.h"

// ---------------------------------------------------------------------------
// Abstract stimulus provider (pulls one input vector each pass)
// ---------------------------------------------------------------------------
class StimulusProvider {
public:
    virtual ~StimulusProvider() = default;
    virtual std::vector<float> next() = 0;   // length == nInput
    virtual double             time() const = 0;
};

// ---------------------------------------------------------------------------
// BrainEngine
// ---------------------------------------------------------------------------
class BrainEngine
{
public:
    BrainEngine(MTL::Device* device,
                uint32_t     nInput,
                uint32_t     nOutput,
                uint32_t     eventsPerPass = 1'000'000);
    ~BrainEngine();                                              // defined in .cpp

    /* model I/O ----------------------------------------------------------- */
    bool load_model(const std::string& filename = "");
    bool save_model(const std::string& filename = "") const;

    /* runtime ------------------------------------------------------------- */
    void set_stimulus(std::shared_ptr<StimulusProvider>);
    std::vector<bool> run_one_pass();        // single synchronous pass
    void start_async();                      // background loop
    void stop_async();
    bool is_running() const { return running_.load(); }

    uint32_t events_per_pass() const { return eventsPerPass_; }

private:
    void build_library_and_queue();

    /* Metal objects */
    MTL::Device*       device_{nullptr};
    MTL::CommandQueue* commandQueue_{nullptr};
    MTL::Library*      defaultLib_{nullptr};

    /* Core */
    std::unique_ptr<Brain>            brain_;
    std::unique_ptr<Logger>           logger_;
    std::shared_ptr<StimulusProvider> stimulus_;

    uint32_t nInput_{0}, nOutput_{0}, eventsPerPass_{0};

    /* Async */
    std::thread       worker_;
    std::atomic<bool> running_{false};
};
