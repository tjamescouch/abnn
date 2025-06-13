//
//  dropout-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#pragma once

#include "layer.h"
#include <Metal/Metal.hpp>
#include <random>


class DropoutLayer : public Layer {
public:
    DropoutLayer(float rate, int inputDim, int featureDim, int batchSize, int sequenceLength);
    ~DropoutLayer() override;

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    void forward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    void backward(MTL::CommandBuffer* cmdBuf, int batchSize) override;

    void updateTargetBufferAt(const float*) override {}
    void updateTargetBufferAt(const float*, int) override {}
    
    
    void setInputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBuffer(BufferType type) override;

    void setOutputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBuffer(BufferType type) override;
    void connectForwardConnections(Layer* previousLayer) override;
    void connectBackwardConnections(Layer* previousLayer) override;
   
    void resetErrors() override;
    
    int inputSize() const override { return inputDim_; }
    int outputSize() const override { return featureDim_; }
    
    void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    
    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;
    
    void setIsTerminal(bool isTerminal) override { isTerminal_ = isTerminal; };
    
    void debugLog() override;
    
private:
    float rate_;
    int sequenceLength_;
    int inputDim_;
    int featureDim_;
    bool isTerminal_;
    int batchSize_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
    
    // New member for CPU-fed randomness
    MTL::Buffer* bufferRandomMask_;
    MTL::Device* _pDevice;
        
    void generateRandomMask();
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

