// brain.cpp  –  ABNN core (CPU ⇄ GPU bridge)

#include "brain.h"

#include <Metal/Metal.hpp>
#include <cassert>
#include <cstring>
#include <random>
#include <iostream>

/*───────────────────────────────────────────────────────────────────────────────
 *  Helpers
 *─────────────────────────────────────────────────────────────────────────────*/
static constexpr uint32_t THREADGROUP_SIZE = 256;   // good default for Apple GPUs

/*───────────────────────────────────────────────────────────────────────────────
 *  ctor / dtor
 *─────────────────────────────────────────────────────────────────────────────*/
Brain::Brain(uint32_t neurons,
             uint32_t synapses,
             uint32_t eventsPerPass)
: N_NRN_(neurons)
, N_SYN_(synapses)
, eventsPerPass_(eventsPerPass)
, hostSynapses_(synapses)
{}

Brain::~Brain()
{
    /*  Metal resources – release only if non-null  */
    auto safeRelease = [](auto*& p) { if (p) { p->release(); p = nullptr; } };

    safeRelease(bufferSynapses_);
    safeRelease(bufferLastFired_);
    safeRelease(bufferLastVisited_);
    safeRelease(bufferClock_);
    safeRelease(bufferRngStates_);
    safeRelease(mcPipeline_);
}

/*───────────────────────────────────────────────────────────────────────────────
 *  Pipeline & buffer allocation
 *─────────────────────────────────────────────────────────────────────────────*/
void Brain::buildPipeline(MTL::Device* device, MTL::Library* library)
{
    /*  Lookup Metal function  */
    auto fn = library->newFunction(NS::String::string("monte_carlo_traversal",
                                                      NS::UTF8StringEncoding));
    assert(fn && "Metal function monte_carlo_traversal not found");

    NS::Error* err = nullptr;
    mcPipeline_ = device->newComputePipelineState(fn, &err);
    assert(mcPipeline_ && "Failed to create compute pipeline");

    fn->release();
}

void Brain::buildBuffers(MTL::Device* device)
{
    /*──────────────── Synapses (SoA packed) ────────────────*/
    size_t synBytes = N_SYN_ * sizeof(SynapsePacked);
    bufferSynapses_ = device->newBuffer(synBytes, MTL::ResourceStorageModeManaged);
    std::memcpy(bufferSynapses_->contents(), hostSynapses_.data(), synBytes);
    bufferSynapses_->didModifyRange(NS::Range(0, synBytes));

    /*──────────────── Per-neuron timing ────────────────*/
    size_t timingBytes = N_NRN_ * sizeof(uint64_t);
    bufferLastFired_   = device->newBuffer(timingBytes, MTL::ResourceStorageModeManaged);
    bufferLastVisited_ = device->newBuffer(timingBytes, MTL::ResourceStorageModeManaged);
    std::memset(bufferLastFired_->contents(),   0, timingBytes);
    std::memset(bufferLastVisited_->contents(), 0, timingBytes);
    bufferLastFired_->didModifyRange  (NS::Range(0, timingBytes));
    bufferLastVisited_->didModifyRange(NS::Range(0, timingBytes));

    /*──────────────── Global clock ────────────────*/
    bufferClock_ = device->newBuffer(sizeof(uint64_t), MTL::ResourceStorageModeManaged);
    *reinterpret_cast<uint64_t*>(bufferClock_->contents()) = 0;
    bufferClock_->didModifyRange(NS::Range(0, sizeof(uint64_t)));

    /*──────────────── RNG seeds (one per event) ────────────────*/
    size_t rngBytes = eventsPerPass_ * sizeof(uint32_t);
    bufferRngStates_ = device->newBuffer(rngBytes, MTL::ResourceStorageModeManaged);
    {
        std::mt19937 rng(42);
        auto* seeds = reinterpret_cast<uint32_t*>(bufferRngStates_->contents());
        for (uint32_t i = 0; i < eventsPerPass_; ++i) seeds[i] = rng();
    }
    bufferRngStates_->didModifyRange(NS::Range(0, rngBytes));
}

/*───────────────────────────────────────────────────────────────────────────────
 *  GPU step  –  advance simulation by eventsPerPass_ events
 *─────────────────────────────────────────────────────────────────────────────*/
void Brain::step(MTL::CommandBuffer* cmdBuf)
{
    assert(mcPipeline_ && "Pipeline not built");
    auto enc = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(mcPipeline_);

    /*  Buffer bindings mirror the kernel signature  */
    enc->setBuffer(bufferSynapses_,    0, 0);
    enc->setBuffer(bufferLastFired_,   0, 1);
    enc->setBuffer(bufferLastVisited_, 0, 2);
    enc->setBuffer(bufferClock_,       0, 3);

    enc->setBytes(&N_SYN_,        sizeof(uint32_t), 4);     // constant uint& N_SYN
    uint32_t tauPrePost = 20'000;                           // ns
    uint32_t tauVisit   = 40'000;                           // ns
    float    alphaLTP   = 0.01f;
    float    alphaLTD   = 0.005f;
    float    wMin       = 0.001f;
    float    wMax       = 1.0f;
    enc->setBytes(&tauPrePost, sizeof(uint32_t), 5);
    enc->setBytes(&tauVisit,   sizeof(uint32_t), 6);
    enc->setBytes(&alphaLTP,   sizeof(float),    7);
    enc->setBytes(&alphaLTD,   sizeof(float),    8);
    enc->setBytes(&wMin,       sizeof(float),    9);
    enc->setBytes(&wMax,       sizeof(float),    10);
    enc->setBuffer(bufferRngStates_, 0, 11);

    /*  Dispatch  */
    uint32_t threads  = eventsPerPass_;
    uint32_t tgWidth  = THREADGROUP_SIZE;
    uint32_t groups   = (threads + tgWidth - 1) / tgWidth;
    MTL::Size tgSize  (tgWidth, 1, 1);
    MTL::Size gridSize(groups * tgWidth, 1, 1);   // full multiples for simplicity
    enc->dispatchThreads(gridSize, tgSize);
    enc->endEncoding();

    /*  Mark buffers modified so CPU view stays coherent (managed mode)  */
    bufferSynapses_->didModifyRange(NS::Range(0, bufferSynapses_->length()));
    bufferLastFired_->didModifyRange(NS::Range(0, bufferLastFired_->length()));
    bufferLastVisited_->didModifyRange(NS::Range(0, bufferLastVisited_->length()));
    bufferClock_->didModifyRange(NS::Range(0, sizeof(uint64_t)));
}

/*───────────────────────────────────────────────────────────────────────────────
 *  CPU reference loop – optional debugging aid
 *─────────────────────────────────────────────────────────────────────────────*/
void Brain::stepCPU(size_t iterations)
{
    uint64_t now     = 0;
    std::vector<uint64_t> lastFired  (N_NRN_, 0);
    std::vector<uint64_t> lastVisited(N_NRN_, 0);
    std::mt19937 rng(123);
    std::uniform_int_distribution<uint32_t> edgeDist(0, N_SYN_ - 1);
    std::uniform_real_distribution<float>   uni(0.0f, 1.0f);

    constexpr uint32_t tauPrePost = 20'000;
    constexpr uint32_t tauVisit   = 40'000;
    constexpr float    alphaLTP   = 0.01f;
    constexpr float    alphaLTD   = 0.005f;

    for (size_t step = 0; step < iterations; ++step, ++now)
    {
        auto& e = hostSynapses_[edgeDist(rng)];
        uint32_t src = e.src, dst = e.dst;
        uint64_t dtSpike = now - lastFired[src];
        uint64_t dtVisit = now - lastVisited[dst];

        float visitFactor = std::exp(-float(dtVisit) / float(tauVisit));
        if (dtSpike < tauPrePost && e.w * visitFactor > uni(rng))
        {
            /*  Fire  */
            lastFired[dst] = now;
            if (dtSpike < tauPrePost)
                e.w += alphaLTP * (1.0f - e.w);
            else
                e.w -= alphaLTD * e.w;
            e.w = std::clamp(e.w, 0.001f, 1.0f);
        }
        lastVisited[dst] = now;
    }
}

/*───────────────────────────────────────────────────────────────────────────────
 *  Save / load
 *─────────────────────────────────────────────────────────────────────────────*/
void Brain::save(std::ostream& os) const
{
    /*  Header  */
    os.write(reinterpret_cast<const char*>(&N_SYN_), sizeof(N_SYN_));
    os.write(reinterpret_cast<const char*>(&N_NRN_), sizeof(N_NRN_));
    /*  Arrays  */
    os.write(reinterpret_cast<const char*>(hostSynapses_.data()),
             hostSynapses_.size() * sizeof(SynapsePacked));
}

void Brain::load(std::istream& is)
{
    is.read(reinterpret_cast<char*>(&N_SYN_), sizeof(N_SYN_));
    is.read(reinterpret_cast<char*>(&N_NRN_), sizeof(N_NRN_));
    hostSynapses_.resize(N_SYN_);
    is.read(reinterpret_cast<char*>(hostSynapses_.data()),
            hostSynapses_.size() * sizeof(SynapsePacked));
}
