//
//  adam-optimizer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-03.
//
#include "adam-optimizer.h"
#include "math-lib.h"

AdamOptimizer::AdamOptimizer(float lr, float beta1, float beta2, float epsilon, uint accumulation_interval) :
bufferGradients_(nullptr),
bufferM_(nullptr),
bufferV_(nullptr),
pipelineState_(nullptr),
timestep_(0),
accumulation_interval_(accumulation_interval),
learningRate_(lr),
beta1_(beta1),
beta2_(beta2),
epsilon_(epsilon) {}

void AdamOptimizer::buildBuffers(MTL::Device* device, size_t paramSize) {
    bufferGradients_ = device->newBuffer(paramSize, MTL::ResourceStorageModeManaged);
    bufferM_ = device->newBuffer(paramSize, MTL::ResourceStorageModeManaged);
    bufferV_ = device->newBuffer(paramSize, MTL::ResourceStorageModeManaged);
    
    memset(bufferGradients_->contents(), 0, paramSize);
    memset(bufferM_->contents(), 0, paramSize);
    memset(bufferV_->contents(), 0, paramSize);
}

void AdamOptimizer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;
    auto adamFunc = library->newFunction(NS::String::string("adam_kernel", NS::UTF8StringEncoding));
    pipelineState_ = device->newComputePipelineState(adamFunc, &error);
    if (!pipelineState_) {
        throw std::runtime_error(error->localizedDescription()->utf8String());
    }
}

void AdamOptimizer::encode(MTL::ComputeCommandEncoder* encoder,
                           MTL::Buffer* params,
                           uint32_t paramCount,
                           uint batchSize) {
    assert(paramCount > 0);
    
    timestep_++;
    bool applyUpdates = (timestep_ % accumulation_interval_) == 0;
    
    encoder->setComputePipelineState(pipelineState_); // <- Must happen first!
    
    const float beta1Scale = 1.0f / (1.0f - pow(beta1_, timestep_));
    const float beta2Scale = 1.0f / (1.0f - pow(beta2_, timestep_));

    // Set buffers and parameters
    encoder->setBuffer(params, 0, 0);
    encoder->setBuffer(bufferGradients_, 0, 1);
    encoder->setBuffer(bufferM_, 0, 2);
    encoder->setBuffer(bufferV_, 0, 3);
    encoder->setBytes(&learningRate_, sizeof(float), 4);
    encoder->setBytes(&beta1_, sizeof(float), 5);
    encoder->setBytes(&beta2_, sizeof(float), 6);
    encoder->setBytes(&epsilon_, sizeof(float), 7);
    encoder->setBytes(&batchSize, sizeof(uint), 8);
    encoder->setBytes(&timestep_, sizeof(uint), 9);
    encoder->setBytes(&paramCount, sizeof(uint), 10);
    encoder->setBytes(&applyUpdates, sizeof(bool), 11);
    encoder->setBytes(&accumulation_interval_, sizeof(uint), 12);
    encoder->setBytes(&beta1Scale, sizeof(float), 13);
    encoder->setBytes(&beta2Scale, sizeof(float), 14);

    // Configure and dispatch threadgroups
    MTL::Size threadgroupSize = MTL::Size(mathlib::min<uint>(paramCount, 1024u), 1, 1);
    MTL::Size threadgroups = MTL::Size((paramCount + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);

    encoder->dispatchThreadgroups(threadgroups, threadgroupSize);
}

MTL::Buffer* AdamOptimizer::gradientBuffer() const {
    return bufferGradients_;
}
