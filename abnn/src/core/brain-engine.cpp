// brain-engine.cpp  –  asynchronous harness driving Brain
// =======================================================

#include "brain-engine.h"
#include <filesystem>
#include <fstream>
#include <mach-o/dyld.h>
#include <limits.h>
#include <iostream>
#include <chrono>
#include <random>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// <Bundle>/Contents/Resources/<filename>  (read-only)
// ─────────────────────────────────────────────────────────────────────────────
static fs::path resource_path(const std::string& filename)
{
    char exePath[PATH_MAX];
    uint32_t len = sizeof(exePath);
    if (_NSGetExecutablePath(exePath, &len) != 0)
        throw std::runtime_error("❌ _NSGetExecutablePath buffer too small");
    fs::path p = fs::canonical(exePath);
    return p.parent_path().parent_path() / "Resources" / filename;
}

// ─────────────────────────────────────────────────────────────────────────────
// ~/Library/Containers/<bundle-id>/Data/<filename>  (read-write)
// ─────────────────────────────────────────────────────────────────────────────
static fs::path data_path(const std::string& filename)
{
    return fs::current_path() / filename;          // sandbox "Data" dir
}

// ─────────────────────────────────────────────────────────────────────────────
// helper : build deterministic Input→Output fan-out + random hidden links
// ─────────────────────────────────────────────────────────────────────────────
static void build_random_graph(Brain& brain, uint32_t nSyn)
{
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> wOut(0.6f, 0.8f);
    std::uniform_real_distribution<float> wHid(0.25f, 0.35f);

    auto* syn = reinterpret_cast<SynapsePacked*>(
                    brain.synapse_buffer()->contents());
    uint32_t idx = 0;

    // dense Input → Output
    for (uint32_t in = 0; in < brain.n_input(); ++in)
        for (uint32_t out = 0; out < brain.n_output() && idx < nSyn; ++out)
            syn[idx++] = { in, brain.n_input() + out, wOut(rng), 0.f };

    // remaining hidden↔hidden
    std::uniform_int_distribution<uint32_t> hid(
        brain.n_input() + brain.n_output(), brain.n_neuron() - 1);
    while (idx < nSyn)
        syn[idx++] = { hid(rng), hid(rng), wHid(rng), 0.f };

    brain.synapse_buffer()->didModifyRange(
        NS::Range(0, nSyn * sizeof(SynapsePacked)));
}

// ─────────────────────────────────────────────────────────────────────────────
BrainEngine::BrainEngine(MTL::Device* device,
                         uint32_t     nInput,
                         uint32_t     nOutput,
                         uint32_t     eventsPerPass)
: device_(device->retain())
, nInput_(nInput)
, nOutput_(nOutput)
, eventsPerPass_(eventsPerPass)
{
    /* 1. Metal queue & library */
    build_library_and_queue();

    /* 2. Create brain core */
    constexpr uint32_t N_HIDDEN = 99;
    constexpr uint32_t N_SYN    = 300;

    brain_ = std::make_unique<Brain>(nInput_, nOutput_,
                                     N_HIDDEN, N_SYN, eventsPerPass_);
    brain_->build_pipeline(device_, defaultLib_);
    brain_->build_buffers (device_);

    /* 3. Logger */
    logger_ = std::make_unique<Logger>(nInput_, nOutput_);

    /* 4. Load or build graph */
    if (!load_model()) {
        std::cout << "🆕  Building random graph…\n";
        build_random_graph(*brain_, N_SYN);
        save_model();        // save to writable container path
    }
}

// ─────────────────────────────────────────────────────────────────────────────
BrainEngine::~BrainEngine()
{
    stop_async();
    if (commandQueue_) commandQueue_->release();
    if (defaultLib_)   defaultLib_->release();
    if (device_)       device_->release();
}

// ─────────────────────────────────────────────────────────────────────────────
void BrainEngine::build_library_and_queue()
{
    commandQueue_ = device_->newCommandQueue();
    defaultLib_   = device_->newDefaultLibrary();
    assert(defaultLib_);
}

// ─────────────────────────────────────────────────────────────────────────────
bool BrainEngine::load_model(const std::string& name)
{
    fs::path fileRW = data_path(name.empty() ? "model.bnn" : name);
    fs::path fileRO = resource_path("model.bnn");
    fs::path file   = fs::exists(fileRW) ? fileRW : fileRO;
    if (!fs::exists(file)) return false;

    std::ifstream is(file, std::ios::binary);
    if (!is) { std::cerr << "❌ cannot open " << file << '\n'; return false; }

    uint32_t syn{}, nrn{};
    is.read(reinterpret_cast<char*>(&syn),4);
    is.read(reinterpret_cast<char*>(&nrn),4);

    uint32_t expect = nInput_ + nOutput_ +
                      (nrn > nInput_ + nOutput_ ? nrn - (nInput_+nOutput_) : 0);
    if (nrn != expect) {
        std::cerr << "⚠️  model neuron-count mismatch → ignore.\n";
        return false;
    }
    is.seekg(0);
    brain_->load(is);
    std::cout << "✅ loaded model (" << (file==fileRW?"Data":"Resources") << ")\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
bool BrainEngine::save_model(const std::string& name) const
{
    fs::path p = data_path(name.empty() ? "model.bnn" : name);
    std::ofstream os(p, std::ios::binary);
    if (!os) { std::cerr << "❌ cannot write " << p << '\n'; return false; }
    brain_->save(os);
    std::cout << "💾 saved model → " << p << '\n';
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
void BrainEngine::set_stimulus(std::shared_ptr<StimulusProvider> s)
{
    stimulus_ = std::move(s);
}

// ─────────────────────────────────────────────────────────────────────────────
std::vector<bool> BrainEngine::run_one_pass()
{
    if (!stimulus_) return {};

    /* 1. inject drive ----------------------------------------------------- */
    std::vector<float> target = stimulus_->next();             // nInput
    brain_->inject_inputs(target, /*Hz*/ 5'000.0f);

    /* 2. GPU traversal ---------------------------------------------------- */
    auto cb = commandQueue_->commandBuffer();
    brain_->encode_traversal(cb);
    cb->commit();
    cb->waitUntilCompleted();

    /* 3. read outputs ----------------------------------------------------- */
    std::vector<bool> b = brain_->read_outputs();
    std::vector<float> pred(nOutput_);
    for (size_t i=0;i<nOutput_; ++i) pred[i] = b[i] ? 1.f : 0.f;

    /* 4. log -------------------------------------------------------------- */
    logger_->log_samples(target, pred);

    static uint32_t step = 0;
    if (++step % 1000 == 0) logger_->flush_to_matlab();

    return b;
}

// ─────────────────────────────────────────────────────────────────────────────
void BrainEngine::start_async()
{
    if (running_.load() || !stimulus_) return;
    running_.store(true);
    worker_ = std::thread([this]{
        while (running_.load()) {
            run_one_pass();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    std::cout << "▶️  Engine async loop started\n";
}

void BrainEngine::stop_async()
{
    if (!running_.load()) return;
    running_.store(false);
    if (worker_.joinable()) worker_.join();
    std::cout << "⏹️  Engine async loop stopped\n";
}
