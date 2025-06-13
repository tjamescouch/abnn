//
//  optimizer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-03.
//
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "common.h"


namespace MTL {
class Device;
class CommandQueue;
class Library;
class CompileOptions;
class CommandBuffer;
class Buffer;
class ComputeCommandEncoder;
}


class Optimizer {
public:
    virtual ~Optimizer() {}
    
    virtual void buildBuffers(MTL::Device* device, size_t paramSize) = 0;
    virtual void buildPipeline(MTL::Device* device, MTL::Library* library) = 0;
    virtual MTL::Buffer* gradientBuffer() const = 0;
    virtual void encode(MTL::ComputeCommandEncoder* encoder,
                        MTL::Buffer* params,
                        uint32_t paramCount,
                        uint batchSize) = 0;
};

#endif
