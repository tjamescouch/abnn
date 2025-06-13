#ifndef MULTI_ATTENTION_LAYER_H
#define MULTI_ATTENTION_LAYER_H

#include "layer.h"
#include "optimizer.h"
#include <Metal/Metal.hpp>
#include <memory>

class MultiHeadAttentionLayer: public Layer {
public:
    MultiHeadAttentionLayer(uint inputDim, uint modelDim, uint seqLength, uint batchSize, uint numHeads);
    ~MultiHeadAttentionLayer();

    void buildBuffers(MTL::Device* device) override;
    void buildPipeline(MTL::Device* device, MTL::Library* library) override;

    void forward(MTL::CommandBuffer* commandBuffer, int batchSize) override;
    void backward(MTL::CommandBuffer* commandBuffer, int batchSize) override;
    
    void resetErrors() override;

    void connectForwardConnections(Layer* previousLayer) override;
    void connectBackwardConnections(Layer* prevLayer) override;

    void setInputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBuffer(BufferType type) override;

    void setOutputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBuffer(BufferType type) override;

    void setIsTerminal(bool isTerminal) override;

    void debugLog() override;
    void onForwardComplete(MTL::CommandQueue*, int) override;
    void onBackwardComplete(MTL::CommandQueue*, int) override;

    void saveParameters(std::ostream&) const override;
    void loadParameters(std::istream&) override;
    
    int inputSize() const override { return inputDim_; }
    int outputSize() const override { return modelDim_; }
    
    void updateTargetBufferAt(const float*) override {}
    void updateTargetBufferAt(const float*, int) override {}
    
    MultiHeadAttentionLayer* setInitializer(std::string initializer) { initializer_ = initializer; return this; }

private:
    uint inputDim_;
    uint modelDim_;
    uint seqLength_;
    uint batchSize_;
    uint numHeads_;
    bool isTerminal_ = false;
    float scale_;
    
    std::string initializer_;

    MTL::Device* device_;

    // Buffers for activations (queries, keys, values)
    MTL::Buffer* bufferQ_ = nullptr;
    MTL::Buffer* bufferK_ = nullptr;
    MTL::Buffer* bufferV_ = nullptr;

    // Input and Output Buffers
    std::unordered_map<BufferType, MTL::Buffer*> inputBuffers_;
    std::unordered_map<BufferType, MTL::Buffer*> outputBuffers_;
    
    MTL::Buffer* bufferAttentionWeights_ = nullptr;
    MTL::Buffer* bufferScratch_ = nullptr;

    // Weight buffers
    MTL::Buffer* weightsQ_ = nullptr;
    MTL::Buffer* weightsK_ = nullptr;
    MTL::Buffer* weightsV_ = nullptr;
    MTL::Buffer* outputProjection_ = nullptr;

    // Optimizers (Adam) - these handle internal gradient buffers
    std::unique_ptr<Optimizer> optimizerWeightsQ_;
    std::unique_ptr<Optimizer> optimizerWeightsK_;
    std::unique_ptr<Optimizer> optimizerWeightsV_;
    std::unique_ptr<Optimizer> optimizerOutputProjection_;

    // Pipeline States
    MTL::ComputePipelineState* forwardPipelineState_ = nullptr;
    MTL::ComputePipelineState* backwardPipelineState_ = nullptr;
};

#endif // SELF_ATTENTION_LAYER_H
