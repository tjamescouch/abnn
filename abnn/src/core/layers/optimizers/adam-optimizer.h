//
//  adam-optimizer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-03.
//
#ifndef ADAM_OPTIMIZER
#define ADAM_OPTIMIZER

#include "optimizer.h"

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(float lr, float beta1, float beta2, float epsilon, uint accumulation_interval);
    
    void buildBuffers(MTL::Device* device, size_t paramSize) override;
    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    
    MTL::Buffer* gradientBuffer() const override;
    
    void encode(MTL::ComputeCommandEncoder* encoder,
                MTL::Buffer* params,
                uint32_t paramCount,
                uint batchSize) override;
    
private:
    MTL::Buffer *bufferGradients_;
    MTL::Buffer *bufferM_, *bufferV_;
    MTL::ComputePipelineState* pipelineState_;
    uint32_t timestep_;
    
    uint accumulation_interval_;
    float learningRate_;
    float beta1_, beta2_, epsilon_;
};

#endif
