#pragma once
/* brain-engine.h  –  asynchronous harness driving Brain
   ===================================================== */

#include <Metal/Metal.hpp>
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <string>
#include "logger.h"

#include "brain.h"

/* ───────── minimal stimulus interface ──────────────────────────────────── */
class StimulusProvider {
public:
    virtual ~StimulusProvider() = default;
    virtual std::vector<float> next()      = 0;   /* returns length nInput   */
    virtual double             time() const= 0;   /* current stimulus time s */
};

/* ───────── BrainEngine ─────────────────────────────────────────────────── */
class BrainEngine
{
public:
    BrainEngine(MTL::Device* device,
                uint32_t     nInput,
                uint32_t     nOutput,
                uint32_t     eventsPerPass = 1'000'000);
    ~BrainEngine();

    /* load / save model (.bnn) ------------------------------------------- */
    bool load_model(const std::string& filename = "");
    bool save_model(const std::string& filename = "") const;

    /* drive / run --------------------------------------------------------- */
    void set_stimulus(std::shared_ptr<StimulusProvider> stim);
    std::vector<bool> run_one_pass();    /* synchronous single pass */

    /* async loop ---------------------------------------------------------- */
    void start_async();
    void stop_async();
    bool is_running() const { return running_.load(); }

    uint32_t events_per_pass() const { return eventsPerPass_; }

private:
    MTL::Device*       device_{nullptr};
    MTL::CommandQueue* commandQueue_{nullptr};
    MTL::Library*      defaultLib_{nullptr};

    std::unique_ptr<Brain>  brain_;
    std::unique_ptr<Logger> logger_;

    std::shared_ptr<StimulusProvider> stim_;

    uint32_t nIn_{0}, nOut_{0}, eventsPerPass_{0};

    std::thread       worker_;
    std::atomic<bool> running_{false};
};
