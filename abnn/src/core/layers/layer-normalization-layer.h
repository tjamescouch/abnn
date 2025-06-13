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

class LayerNormalizationLayer : public Layer {
public:
    LayerNormalizationLayer(int featureDim, int seqLength, int batchSize, float learningRate, float epsilon);
    ~LayerNormalizationLayer() override;

    void buildBuffers(MTL::Device* device) override;
    void buildPipeline(MTL::Device* device, MTL::Library* library) override;

    void forward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    void backward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    
    void setInputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBuffer(BufferType type) override;

    void setOutputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBuffer(BufferType type) override;
    
    void resetErrors() override;
    
    int inputSize() const override { return featureDim_; }
    int outputSize() const override { return featureDim_; }

    void updateTargetBufferAt(const float* targetData) override;
    void updateTargetBufferAt(const float* targetData, int batchSize) override;

    void connectForwardConnections(Layer* previousLayer) override;
    void connectBackwardConnections(Layer* previousLayer) override;

    void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;

    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;

    void debugLog() override { /*TODO*/ }

    void setIsTerminal(bool isTerminal) override { isTerminal_ = isTerminal; }
    
private:
    int featureDim_;
    int seqLength_;
    float epsilon_;
    bool isTerminal_;
    int batchSize_;
    float learningRate_;

    // Parameter buffers
    MTL::Buffer* bufferGamma_; // Scale parameter
    MTL::Buffer* bufferBeta_;  // Shift parameter
    MTL::Buffer* bufferDebug_;

    // Intermediate per-sample statistics buffers
    MTL::Buffer* bufferSavedMean_;
    MTL::Buffer* bufferSavedVariance_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;

    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
    
    std::unique_ptr<Optimizer> optimizerGamma_;
    std::unique_ptr<Optimizer> optimizerBeta_;

    void initializeParameters(MTL::Device* device);
};
