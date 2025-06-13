//
//  reshape-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#ifndef RESHAPE_LAYER_H
#define RESHAPE_LAYER_H

#include "layer.h"

class ReshapeLayer : public Layer {
public:
    ReshapeLayer(int sequenceLength, int inputSize, int outputSize, int batchSize);
    ~ReshapeLayer();

    void forward(MTL::CommandBuffer* commandBuffer, int batchSize) override;
    void backward(MTL::CommandBuffer* commandBuffer, int batchSize) override;

    void buildBuffers(MTL::Device* device) override;
    void buildPipeline(MTL::Device* device, MTL::Library* library) override;

    void setInputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBuffer(BufferType type) override;
    void setOutputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBuffer(BufferType type) override;
    
    void resetErrors() override;

    int inputSize() const override;
    int outputSize() const override;

    void updateTargetBufferAt(const float* targetData) override;
    void updateTargetBufferAt(const float* targetData, int batchSize) override;

    void connectForwardConnections(Layer* previousLayer) override;
    void connectBackwardConnections(Layer* previousLayer) override;

    void debugLog() override;
    void onForwardComplete(MTL::CommandQueue* commandQueue, int batchSize) override;
    void onBackwardComplete(MTL::CommandQueue* commandQueue, int batchSize) override;

    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;

    void setIsTerminal(bool isTerminal) override;

private:
    /* Remove any not needed member variables */
    int sequenceLength_;
    int inputSize_;
    int outputSize_;
    int batchSize_;
    bool isTerminal_;

    std::unordered_map<BufferType, MTL::Buffer*> inputBuffers_;
    std::unordered_map<BufferType, MTL::Buffer*> outputBuffers_;
    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
};

#endif
