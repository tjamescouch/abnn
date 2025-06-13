//
//  map-reduce-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-04.
//

#ifndef MAP_REDUCE_LAYER_H
#define MAP_REDUCE_LAYER_H

#include "layer.h"

class MapReduceLayer : public Layer {
public:
    MapReduceLayer(int inputSize, int outputSize, ReductionType reductionType);
    ~MapReduceLayer();

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    
    void forward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    void backward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    
    void resetErrors() override;
    
    void connectForwardConnections(Layer* previousLayer) override;
    void connectBackwardConnections(Layer* previousLayer) override;
    
    void setInputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBuffer(BufferType type) override;
    void setOutputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBuffer(BufferType type) override;

    int inputSize() const override;
    int outputSize() const override;

    void updateTargetBufferAt(const float* targetData) override;
    void updateTargetBufferAt(const float* targetData, int batchSize) override;

    void debugLog() override;

    void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;

    void saveParameters(std::ostream& outStream) const override;
    void loadParameters(std::istream& inStream) override;

    void setIsTerminal(bool isTerminal) override;

private:
    int output_dim_;
    int inputSize_;
    ReductionType reductionType_;
    bool isTerminal_;
    
    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    // Explicit mapping of BufferType to buffer arrays
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
};

#endif
