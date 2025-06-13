//
//  layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-02-27.
//

#ifndef LAYER_H
#define LAYER_H

#include <functional>
#include "common.h"

// Forward declarations for Metal types.
namespace MTL {
class Buffer;
class Device;
class Library;
class CommandBuffer;
class CommandQueue;
class ComputePipelineState;
}

enum class ActivationFunction {
    Linear = 0,
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
    GELU
};

inline ActivationFunction parseActivation(const std::string& activation) {
    if (activation == "linear") return ActivationFunction::Linear;
    if (activation == "relu") return ActivationFunction::ReLU;
    if (activation == "tanh") return ActivationFunction::Tanh;
    if (activation == "sigmoid") return ActivationFunction::Sigmoid;
    if (activation == "softmax") return ActivationFunction::Softmax;
    if (activation == "gelu") return ActivationFunction::GELU;
    throw std::invalid_argument("Unknown activation: " + activation);
}

enum class ReductionType {
    Sum = 0,
    Mean,
    Max,
    Min,
    Softmax,
};

inline ReductionType parseReductionType(const std::string& reductionType) {
    if (reductionType == "sum") return ReductionType::Sum;
    if (reductionType == "mean") return ReductionType::Mean;
    if (reductionType == "max") return ReductionType::Max;
    if (reductionType == "min") return ReductionType::Min;
    if (reductionType == "softmax") return ReductionType::Softmax;
    throw std::invalid_argument("Unknown reduction type: " + reductionType);
}

enum class BufferType : unsigned int {
    Input = 0,
    HiddenState,
    PrevHiddenState,
    Output,
    Debug,
    Targets,
    HiddenErrors,
    Gradients,
    Activation,
    OutgoingErrors,
    IncomingErrors,
    Delta,
};

class Layer {
public:
    virtual ~Layer() {}
    // Called to build pipeline states from the given device and library.
    virtual void buildPipeline(MTL::Device* device, MTL::Library* library) = 0;
    // Called to allocate any buffers needed for the layer.
    virtual void buildBuffers(MTL::Device* device) = 0;
    // Record commands for the forward pass.
    virtual void forward(MTL::CommandBuffer* cmdBuf, int batchSize) = 0;
    // Record commands for the backward pass.
    virtual void backward(MTL::CommandBuffer* cmdBuf, int batchSize) = 0;
    
    virtual void setInputBuffer(BufferType type, MTL::Buffer* buffer) = 0;
    virtual MTL::Buffer* getInputBuffer(BufferType type) = 0;
    
    virtual void setOutputBuffer(BufferType type, MTL::Buffer* buffer) = 0;
    virtual MTL::Buffer* getOutputBuffer(BufferType type) = 0;
    
    virtual void resetErrors() = 0;
    
    virtual int inputSize() const = 0;
    virtual int outputSize() const = 0;
    
    virtual void updateTargetBufferAt(const float* targetData) = 0;
    virtual void updateTargetBufferAt(const float* targetData,
                                      int batchSize) = 0;
    
    virtual void connectForwardConnections(Layer* previousLayer) = 0;
    virtual void connectBackwardConnections(Layer* previousLayer) = 0;
    
    virtual void debugLog() = 0;
    virtual void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) = 0;
    virtual void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSizee) = 0;
    
    virtual void saveParameters(std::ostream& os) const = 0;
    virtual void loadParameters(std::istream& is) = 0;
    
    virtual void setIsTerminal(bool isTerminal) = 0;
    
protected:
    std::string name_;
    
public:
    void setName(const std::string& name) { name_ = name; }
    const std::string& getName() const { return name_; }
    
}; 

#endif // LAYER_H
