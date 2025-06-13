//
//  residual-connection-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#ifndef RESIDUAL_CONNECTION_LAYER_H
#define RESIDUAL_CONNECTION_LAYER_H

#include "layer.h"

class ResidualConnectionLayer : public Layer {
public:
    ResidualConnectionLayer(int featureDim, int sequenceLength, int batchSize, float residualScale);
    ~ResidualConnectionLayer();

    void forward(MTL::CommandBuffer* commandBuffer, int batchSize) override;
    void backward(MTL::CommandBuffer* commandBuffer, int batchSize) override;

    void buildBuffers(MTL::Device* device) override;
    void buildPipeline(MTL::Device* device, MTL::Library* library) override;

    ResidualConnectionLayer* setFromLayer(Layer* fromLayer);

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
    int sequenceLength_;
    int featureDim_;
    int batchSize_;
    bool isTerminal_;
    float residualScale_;

    Layer* fromLayer_;
    std::unordered_map<BufferType, MTL::Buffer*> inputBuffers_;
    std::unordered_map<BufferType, MTL::Buffer*> outputBuffers_;
    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
};

#endif // RESIDUAL_CONNECTION_LAYER_H
