//
//  batch-normalization-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#pragma once
#include <Metal/Metal.hpp>
#include <vector>
#include "layer.h"
#include "optimizer.h"

class BatchNormalizationLayer : public Layer {
public:
    BatchNormalizationLayer(int inputDim, int outputDim, int batchSize, int sequenceLength, float learningRate, float epsilon = 1e-5f);
    ~BatchNormalizationLayer() override;

    void buildBuffers(MTL::Device* device) override;
    void buildPipeline(MTL::Device* device, MTL::Library* library) override;

    void forward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    void backward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    
    void setInputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBuffer(BufferType type) override;

    void setOutputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBuffer(BufferType type) override;
    
    void resetErrors() override;
    
    int inputSize() const override { return inputDim_; }
    int outputSize() const override;
    void updateTargetBufferAt(const float* targetData) override;
    void updateTargetBufferAt(const float* targetData, int batchSize) override;
    
    void connectForwardConnections(Layer* previousLayer) override;
    void connectBackwardConnections(Layer* previousLayer) override;
    
    void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    
    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;

    void debugLog() override {/*TODO*/}
        
    void setIsTerminal(bool isTerminal) override { isTerminal_ = isTerminal; };
    
private:
    uint inputDim_;
    uint outputDim_;
    int sequenceLength_;
    float epsilon_;
    bool isTerminal_;
    uint batchSize_;
    size_t bufferSize_;
    float learningRate_;

    // Parameter buffers
    MTL::Buffer* bufferGamma_; // Scale
    MTL::Buffer* bufferBeta_;  // Shift
    MTL::Buffer* bufferDebug_;
    
    std::unique_ptr<Optimizer> optimizerGamma_;
    std::unique_ptr<Optimizer> optimizerBeta_;

    // Running averages for inference
    MTL::Buffer* bufferRunningMean_;
    MTL::Buffer* bufferRunningVariance_;
    MTL::Buffer* bufferSavedMean_;
    MTL::Buffer* bufferSavedVariance_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;

    void initializeParameters(MTL::Device* device);
};
