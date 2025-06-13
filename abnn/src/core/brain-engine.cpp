// brain-engine.cpp  –  Minimal harness for the ABNN “Brain”

#include "brain-engine.h"

#include <fstream>
#include <cassert>
#include <iostream>
#include <mach-o/dyld.h>
#include <filesystem>
#include "logger.h"

using namespace std;

/*───────────────────────────────────────────────────────────────────────────────
 * ctor / dtor
 *─────────────────────────────────────────────────────────────────────────────*/
BrainEngine::BrainEngine(MTL::Device* device,
                         uint32_t     eventsPerPass)
: device_(device->retain())
, eventsPerPass_(eventsPerPass)
, commandQueue_(device_->newCommandQueue())
{
    buildMetalObjects();
}

BrainEngine::~BrainEngine()
{
    if (commandQueue_)   commandQueue_->release();
    if (computeLibrary_) computeLibrary_->release();
    if (device_)         device_->release();
    if (renormPipeline_) renormPipeline_->release();
}

/*───────────────────────────────────────────────────────────────────────────────
 * Metal setup
 *─────────────────────────────────────────────────────────────────────────────*/
void BrainEngine::buildMetalObjects()
{
    namespace fs = std::filesystem;
    
    computeLibrary_ = device_->newDefaultLibrary();
    assert(computeLibrary_ && "Failed to load default .metallib");

    /*  Compile renormalisation pipeline once  */
    auto fn = computeLibrary_->newFunction(
        NS::String::string("renormalise_clock_and_times", NS::UTF8StringEncoding));
    assert(fn && "missing renormalise_clock_and_times() in brain.metal");

    NS::Error* err = nullptr;
    renormPipeline_ = device_->newComputePipelineState(fn, &err);
    assert(renormPipeline_ && "failed to create renorm pipeline");
    fn->release();
    
    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) {
        throw std::runtime_error("❌ Executable path buffer too small.");
    }
    
    fs::path executablePath = fs::canonical(path);
    fs::path resourcePath = executablePath.parent_path().parent_path() / "Resources" / "model.bnn";
    
    this->loadModel(resourcePath);
}

/*───────────────────────────────────────────────────────────────────────────────
 * Model I/O
 *─────────────────────────────────────────────────────────────────────────────*/
bool BrainEngine::loadModel(const string& path)
{
    ifstream is(path, ios::binary);
    if (!is) { cerr << "❌  Could not open " << path << endl; return false; }

    /*  Peek at the header to know graph size  */
    uint32_t nSyn=0, nNrn=0;
    is.read(reinterpret_cast<char*>(&nSyn), sizeof(nSyn));
    is.read(reinterpret_cast<char*>(&nNrn), sizeof(nNrn));
    if (!is) { cerr << "❌  Corrupt header in " << path << endl; return false; }
    is.seekg(0);                              // rewind

    /*  Build Brain object if needed / shape changed  */
    brain_ = make_unique<Brain>(nNrn, nSyn, eventsPerPass_);
    brain_->buildPipeline(device_, computeLibrary_);
    brain_->buildBuffers(device_);
    brain_->load(is);
    buffersBuilt_ = true;

    cout << "✅  Loaded model '" << path
         << "'  (neurons=" << nNrn << ", synapses=" << nSyn << ")\n";
    return true;
}

bool BrainEngine::saveModel(const string& path) const
{
    if (!brain_) { cerr << "❌  No brain to save\n"; return false; }
    ofstream os(path, ios::binary);
    if (!os) { cerr << "❌  Could not open " << path << " for writing\n"; return false; }
    brain_->save(os);
    cout << "💾  Saved model to " << path << endl;
    return true;
}

/*───────────────────────────────────────────────────────────────────────────────
 * Run network
 *─────────────────────────────────────────────────────────────────────────────*/
void BrainEngine::run(uint32_t passes)
{
    assert(buffersBuilt_ && "loadModel() needs to be called first");

    uint32_t hostTick = 0;   // mirror of GPU clock (32-bit, wraps like GPU)

    Logger::log << "🔋 Running network" << std::endl;
    for (uint32_t p = 0; p < passes; ++p)
    {
        auto cmdBuf = commandQueue_->commandBuffer();

        /* 1) Monte-Carlo traversal for `eventsPerPass_` events */
        brain_->step(cmdBuf);
        hostTick += eventsPerPass_;           // mirror the GPU increment

        /* 2) If we’re near wrap, append renorm kernel */
        if (hostTick > Brain::kRenormThreshold)
        {
            ensureRenormalised(cmdBuf);       // encode kernel
            hostTick = 0;                     // host mirror also reset
        }

        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
    }
}

/*───────────────────────────────────────────────────────────────────────────────
 * Append renormalisation kernel to a command buffer
 *─────────────────────────────────────────────────────────────────────────────*/
void BrainEngine::ensureRenormalised(MTL::CommandBuffer* cmdBuf)
{
    auto enc = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(renormPipeline_);

    enc->setBuffer(brain_->lastFiredBuffer(),   0, 0);
    enc->setBuffer(brain_->lastVisitedBuffer(), 0, 1);
    enc->setBuffer(brain_->clockBuffer(),       0, 2);
    uint32_t nNeurons = brain_->neuronCount();
    enc->setBytes(&nNeurons, sizeof(uint32_t),  3);

    /* Thread-grid: one thread per neuron */
    const uint32_t tgWidth = 256;
    MTL::Size tgSize(tgWidth,1,1);
    MTL::Size grid((nNeurons + tgWidth - 1) / tgWidth * tgWidth, 1,1);
    enc->dispatchThreads(grid, tgSize);
    enc->endEncoding();
}
