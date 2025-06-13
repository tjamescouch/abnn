// brain.cpp  –  core event-driven ABNN implementation
// ===================================================

#include "brain.h"

#include <Metal/Metal.hpp>
#include <cassert>
#include <cstring>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
// ctor / dtor
// ─────────────────────────────────────────────────────────────────────────────
Brain::Brain(uint32_t nInput,
             uint32_t nOutput,
             uint32_t nHidden,
             uint32_t nSynapses,
             uint32_t eventsPerPass)
: N_INPUT_(nInput)
, N_OUTPUT_(nOutput)
, N_HIDDEN_(nHidden)
, N_NRN_(nInput + nOutput + nHidden)
, N_SYN_(nSynapses)
, EVENTS_(eventsPerPass)
, hostSyn_(nSynapses)
{}

Brain::~Brain() { release_all(); }

// ─────────────────────────────────────────────────────────────────────────────
void Brain::release_all()
{
    auto rel = [](auto*& p){ if (p){ p->release(); p=nullptr; } };
    rel(bufSyn_); rel(bufLastFire_); rel(bufLastVisit_);
    rel(bufClock_); rel(bufRng_);
    rel(pipeTraverse_); rel(pipeRenorm_);
}

// ─────────────────────────────────────────────────────────────────────────────
void Brain::build_pipeline(MTL::Device* dev, MTL::Library* lib)
{
    NS::Error* err = nullptr;
    auto fTrav = lib->newFunction(
        NS::String::string("monte_carlo_traversal", NS::UTF8StringEncoding));
    pipeTraverse_ = dev->newComputePipelineState(fTrav, &err);
    fTrav->release();

    auto fRen = lib->newFunction(
        NS::String::string("renormalise_clock_and_times", NS::UTF8StringEncoding));
    pipeRenorm_ = dev->newComputePipelineState(fRen, &err);
    fRen->release();
}

// ─────────────────────────────────────────────────────────────────────────────
void Brain::build_buffers(MTL::Device* dev)
{
    // Synapses in managed mode (GPU-only after init)
    size_t synBytes = N_SYN_ * sizeof(SynapsePacked);
    bufSyn_ = dev->newBuffer(synBytes, MTL::ResourceStorageModeManaged);
    std::memset(bufSyn_->contents(), 0, synBytes);

    // Per-neuron times in SHARED mode for free CPU-GPU coherence
    size_t nBytes = N_NRN_ * sizeof(uint32_t);
    bufLastFire_  = dev->newBuffer(nBytes, MTL::ResourceStorageModeShared);
    bufLastVisit_ = dev->newBuffer(nBytes, MTL::ResourceStorageModeShared);
    std::memset(bufLastFire_->contents(),  0, nBytes);
    std::memset(bufLastVisit_->contents(), 0, nBytes);

    // Global clock in SHARED mode
    bufClock_ = dev->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    *reinterpret_cast<uint32_t*>(bufClock_->contents()) = 0;

    // RNG seeds (managed, GPU-only)
    bufRng_ = dev->newBuffer(EVENTS_ * sizeof(uint32_t),
                             MTL::ResourceStorageModeManaged);
    std::mt19937 rng(123);
    uint32_t* p = reinterpret_cast<uint32_t*>(bufRng_->contents());
    for (uint32_t i=0;i<EVENTS_;++i) p[i] = rng();
    bufRng_->didModifyRange(NS::Range(0, EVENTS_*sizeof(uint32_t)));
}

// ─────────────────────────────────────────────────────────────────────────────
void Brain::encode_traversal(MTL::CommandBuffer* cb)
{
    auto enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(pipeTraverse_);

    enc->setBuffer(bufSyn_,       0, 0);
    enc->setBuffer(bufLastFire_,  0, 1);
    enc->setBuffer(bufLastVisit_, 0, 2);
    enc->setBuffer(bufClock_,     0, 3);

    enc->setBytes(&N_SYN_, sizeof(uint32_t), 4);

    uint32_t tauVis = 4000, tauPre = 2000;
    enc->setBytes(&tauVis, sizeof(uint32_t), 5);
    enc->setBytes(&tauPre, sizeof(uint32_t), 6);

    float aLTP = 0.01f, aLTD = 0.005f, wMin = 0.001f, wMax = 1.0f;
    enc->setBytes(&aLTP, sizeof(float), 7);
    enc->setBytes(&aLTD, sizeof(float), 8);
    enc->setBytes(&wMin, sizeof(float), 9);
    enc->setBytes(&wMax, sizeof(float),10);

    enc->setBuffer(bufRng_, 0, 11);

    const uint tg = 256;
    MTL::Size tgSize(tg,1,1);
    MTL::Size grid(((EVENTS_+tg-1)/tg)*tg,1,1);
    enc->dispatchThreads(grid, tgSize);
    enc->endEncoding();

    renormalise_if_needed(cb);
}

// ─────────────────────────────────────────────────────────────────────────────
void Brain::renormalise_if_needed(MTL::CommandBuffer* cb)
{
    uint32_t now = *reinterpret_cast<uint32_t*>(bufClock_->contents());
    if (now <= kRenormThreshold) return;

    auto enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(pipeRenorm_);
    enc->setBuffer(bufLastFire_,   0, 0);
    enc->setBuffer(bufLastVisit_,  0, 1);
    enc->setBuffer(bufClock_,      0, 2);
    enc->setBytes(&N_NRN_, sizeof(uint32_t), 3);

    const uint tg = 256;
    MTL::Size tgSize(tg,1,1);
    MTL::Size grid(((N_NRN_+tg-1)/tg)*tg,1,1);
    enc->dispatchThreads(grid, tgSize);
    enc->endEncoding();
}

// ─────────────────────────────────────────────────────────────────────────────
void Brain::inject_inputs(const std::vector<float>& analogue, float poissonHz)
{
    assert(analogue.size() == N_INPUT_);
    float pTick = poissonHz * kTickNS * 1e-9f;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> uni(0.f,1.f);

    uint32_t* last = reinterpret_cast<uint32_t*>(bufLastFire_->contents());
    uint32_t  now  = *reinterpret_cast<uint32_t*>(bufClock_->contents());

    for (uint32_t i=0;i<N_INPUT_;++i)
        if (uni(rng) < pTick * analogue[i])
            last[i] = now;
}

// ─────────────────────────────────────────────────────────────────────────────
std::vector<bool> Brain::read_outputs() const
{
    std::vector<bool> fired(N_OUTPUT_, false);

    const uint32_t* last = reinterpret_cast<const uint32_t*>(bufLastFire_->contents());
    uint32_t now = *reinterpret_cast<const uint32_t*>(bufClock_->contents());

    uint32_t window = EVENTS_ * 10;                // 10-pass window
    uint32_t start  = now >= window ? now - window : 0;

    for (uint32_t o=0;o<N_OUTPUT_;++o) {
        uint32_t ts = last[N_INPUT_ + o];
        if (ts >= start && ts < now) fired[o] = true;
    }
    return fired;
}

// ─────────────────────────────────────────────────────────────────────────────
// save / load (unchanged)
// ─────────────────────────────────────────────────────────────────────────────
void Brain::save(std::ostream& os) const
{
    os.write(reinterpret_cast<const char*>(&N_SYN_), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&N_NRN_), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(bufSyn_->contents()),
             N_SYN_ * sizeof(SynapsePacked));
}

void Brain::load(std::istream& is)
{
    uint32_t syn=0,nrn=0;
    is.read(reinterpret_cast<char*>(&syn),4);
    is.read(reinterpret_cast<char*>(&nrn),4);
    assert(syn==N_SYN_ && nrn==N_NRN_);
    is.read(reinterpret_cast<char*>(bufSyn_->contents()),
            N_SYN_*sizeof(SynapsePacked));
    bufSyn_->didModifyRange(NS::Range(0, N_SYN_*sizeof(SynapsePacked)));
}
