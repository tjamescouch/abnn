//
//  residual-connection-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include "self-attention-layer.h"
#include "logger.h"
#include "configuration-manager.h"
#include "model-config.h"
#include "adam-optimizer.h"
#include "weight-initializer.h"

SelfAttentionLayer::SelfAttentionLayer(uint inputDim, uint modelDim, uint seqLength, uint batchSize)
    : inputDim_(inputDim),
      modelDim_(modelDim),
      seqLength_(seqLength),
      batchSize_(batchSize),
      device_(nullptr),
      bufferAttentionWeights_(nullptr),
      bufferScratch_(nullptr),
      bufferQ_(nullptr),
      bufferK_(nullptr),
      bufferV_(nullptr),
      weightsQ_(nullptr),
      weightsK_(nullptr),
      weightsV_(nullptr),
      outputProjection_(nullptr),
      optimizerWeightsQ_(nullptr),
      optimizerWeightsK_(nullptr),
      optimizerWeightsV_(nullptr),
      optimizerOutputProjection_(nullptr),
      forwardPipelineState_(nullptr),
      backwardPipelineState_(nullptr) {
}

SelfAttentionLayer::~SelfAttentionLayer() {
    if (bufferAttentionWeights_) bufferAttentionWeights_->release();
    if (bufferScratch_) bufferScratch_->release();
    
    if (bufferQ_) bufferQ_->release();
    if (bufferK_) bufferK_->release();
    if (bufferV_) bufferV_->release();

    if (weightsQ_) weightsQ_->release();
    if (weightsK_) weightsK_->release();
    if (weightsV_) weightsV_->release();
    if (outputProjection_) outputProjection_->release();

    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void SelfAttentionLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;

    auto kernelFn = library->newFunction(NS::String::string("forward_self_attention", NS::ASCIIStringEncoding));
    forwardPipelineState_ = device->newComputePipelineState(kernelFn, &error);
    kernelFn->release();

    if (!forwardPipelineState_) {
        Logger::log << "Error occurred creating forward pipeline: " << error->localizedDescription()->utf8String() << std::endl;
        assert(false);
    }

    auto kernelBackwardFn = library->newFunction(NS::String::string("backward_self_attention", NS::ASCIIStringEncoding));
    backwardPipelineState_ = device->newComputePipelineState(kernelBackwardFn, &error);
    kernelBackwardFn->release();

    if (!backwardPipelineState_) {
        Logger::log << "Error occurred creating self_attention_backward pipeline: " << error->localizedDescription()->utf8String() << std::endl;
        std::exit(-1);
    }
    
    ModelConfig* pConfig = ConfigurationManager::instance().getConfig();
    auto parameters = pConfig->training.optimizer.parameters;
    auto optimizerConfig = pConfig->training.optimizer;
    
    uint accumulation_interval = optimizerConfig.accumulation_interval;
    float lr = optimizerConfig.learning_rate; //FIXME - get layer specific value in factory
    float beta1 = optimizerConfig.beta1;
    float beta2 = optimizerConfig.beta2;
    float epsilon = optimizerConfig.epsilon;
        
    optimizerWeightsQ_         = std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon, accumulation_interval);
    optimizerWeightsK_         = std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon, accumulation_interval);
    optimizerWeightsV_         = std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon, accumulation_interval);
    optimizerOutputProjection_ = std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon, accumulation_interval);
    
    optimizerWeightsQ_->buildPipeline(device, library);
    optimizerWeightsK_->buildPipeline(device, library);
    optimizerWeightsV_->buildPipeline(device, library);
    optimizerOutputProjection_->buildPipeline(device, library);
}

void SelfAttentionLayer::buildBuffers(MTL::Device* device) {
    const size_t attentionBufferSize = batchSize_ * seqLength_ * seqLength_ * sizeof(float);
    const size_t activationBufferSize = seqLength_ * modelDim_ * sizeof(float);
    const size_t weightsBufferSize = inputDim_ * modelDim_ * sizeof(float);
    const size_t scratchPerThreadSize = 3*modelDim_ + 2*seqLength_ + 2*(seqLength_*modelDim_);
    const size_t scratchBufferSize = batchSize_ * seqLength_ * scratchPerThreadSize * sizeof(float);
    
    bufferAttentionWeights_ = device->newBuffer(attentionBufferSize, MTL::ResourceStorageModeManaged);
    memset(bufferAttentionWeights_->contents(), 0, attentionBufferSize);
    
    bufferScratch_ = device->newBuffer(scratchBufferSize, MTL::ResourceStorageModeManaged);
    memset(bufferScratch_->contents(), 0, scratchBufferSize);

    // Buffers for projected queries (Q), keys (K), and values (V)
    bufferQ_ = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);
    bufferK_ = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);
    bufferV_ = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);
    memset(bufferQ_->contents(), 0, activationBufferSize);
    memset(bufferK_->contents(), 0, activationBufferSize);
    memset(bufferV_->contents(), 0, activationBufferSize);

    // Buffers for weights of Q, K, V projections and the output projection
    weightsQ_ = device->newBuffer(weightsBufferSize, MTL::ResourceStorageModeManaged);
    weightsK_ = device->newBuffer(weightsBufferSize, MTL::ResourceStorageModeManaged);
    weightsV_ = device->newBuffer(weightsBufferSize, MTL::ResourceStorageModeManaged);
    outputProjection_ = device->newBuffer(weightsBufferSize, MTL::ResourceStorageModeManaged);
    
    float* q = static_cast<float*>(weightsQ_->contents());
    float* k = static_cast<float*>(weightsK_->contents());
    float* v = static_cast<float*>(weightsV_->contents());
    float* o = static_cast<float*>(outputProjection_->contents());
    if (initializer_ == "he") {
        WeightInitializer::initializeHe(q, inputDim_, modelDim_);
        WeightInitializer::initializeHe(k, inputDim_, modelDim_);
        WeightInitializer::initializeHe(v, inputDim_, modelDim_);
        WeightInitializer::initializeHe(o, inputDim_, modelDim_);
    } else {
        WeightInitializer::initializeXavier(q, inputDim_, modelDim_);
        WeightInitializer::initializeXavier(k, inputDim_, modelDim_);
        WeightInitializer::initializeXavier(v, inputDim_, modelDim_);
        WeightInitializer::initializeXavier(o, inputDim_, modelDim_);
    }
    weightsQ_->didModifyRange(NS::Range(0, weightsQ_->length()));
    weightsK_->didModifyRange(NS::Range(0, weightsK_->length()));
    weightsV_->didModifyRange(NS::Range(0, weightsV_->length()));
    outputProjection_->didModifyRange(NS::Range(0, outputProjection_->length()));


    outputBuffers_[BufferType::Output] = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::OutgoingErrors] = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);
    
    optimizerWeightsQ_->buildBuffers(device, weightsBufferSize);
    optimizerWeightsK_->buildBuffers(device, weightsBufferSize);
    optimizerWeightsV_->buildBuffers(device, weightsBufferSize);
    optimizerOutputProjection_->buildBuffers(device, weightsBufferSize);
}

void SelfAttentionLayer::forward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    
    uint bs = (uint)batchSize;

    // Binding input and weight buffers
    encoder->setBuffer(inputBuffers_[BufferType::Input], 0, 0);
    encoder->setBuffer(weightsQ_, 0, 1);
    encoder->setBuffer(weightsK_, 0, 2);
    encoder->setBuffer(weightsV_, 0, 3);
    encoder->setBuffer(outputProjection_, 0, 4);

    // Binding buffers for intermediate Q, K, V projections
    encoder->setBuffer(bufferQ_, 0, 5);
    encoder->setBuffer(bufferK_, 0, 6);
    encoder->setBuffer(bufferV_, 0, 7);

    // Binding the final output buffer
    encoder->setBuffer(outputBuffers_[BufferType::Output], 0, 8);

    // Constant parameters (dimensions)
    encoder->setBytes(&bs, sizeof(uint), 9);
    encoder->setBytes(&seqLength_, sizeof(uint), 10);
    encoder->setBytes(&inputDim_, sizeof(uint), 11);
    encoder->setBytes(&modelDim_, sizeof(uint), 12);

    // Thread dispatch configuration
    const int gridSize = batchSize * seqLength_ * modelDim_;
    MTL::Size threadsPerGrid = MTL::Size(gridSize, 1, 1);
    MTL::Size threadsPerGroup = MTL::Size(std::min(gridSize, 1024), 1, 1);

    encoder->dispatchThreads(threadsPerGrid, threadsPerGroup);
    encoder->endEncoding();
}

void SelfAttentionLayer::backward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);
    uint bs = (uint)batchSize;

    // Binding buffers (must match exactly kernel buffer indices)
    encoder->setBuffer(inputBuffers_[BufferType::Input], 0, 0);                  // Inputs from forward pass
    encoder->setBuffer(weightsQ_, 0, 1);                                         // Q weights
    encoder->setBuffer(weightsK_, 0, 2);                                         // K weights
    encoder->setBuffer(weightsV_, 0, 3);                                         // V weights
    encoder->setBuffer(outputProjection_, 0, 4);                                 // Output projection weights
    
    encoder->setBuffer(bufferQ_, 0, 5);
    encoder->setBuffer(bufferK_, 0, 6);
    encoder->setBuffer(bufferV_, 0, 7);
    encoder->setBuffer(bufferAttentionWeights_, 0, 8);

    encoder->setBuffer(outputBuffers_[BufferType::OutgoingErrors], 0, 9);          // Errors leaving the layer
    encoder->setBuffer(inputBuffers_[BufferType::IncomingErrors], 0, 10);            // Errors entering the layer


    encoder->setBuffer(optimizerWeightsQ_->gradientBuffer(), 0, 11);              // Gradients for weightsQ
    encoder->setBuffer(optimizerWeightsK_->gradientBuffer(), 0, 12);              // Gradients for weightsK
    encoder->setBuffer(optimizerWeightsV_->gradientBuffer(), 0, 13);              // Gradients for weightsV
    encoder->setBuffer(optimizerOutputProjection_->gradientBuffer(), 0, 14);     // Gradients for outputProjection

    // Constant arguments (dimensions)
    encoder->setBytes(&bs, sizeof(uint), 15);
    encoder->setBytes(&seqLength_, sizeof(uint), 16);
    encoder->setBytes(&inputDim_, sizeof(uint), 17);
    encoder->setBytes(&modelDim_, sizeof(uint), 18);
    
    //Scratch
    encoder->setBuffer(bufferScratch_, 0, 19);     // Gradients for outputProjection

    // Thread dispatch configuration
    const int gridSize = batchSize * seqLength_ * modelDim_;
    MTL::Size threadsPerGrid = MTL::Size(gridSize, 1, 1);
    MTL::Size threadsPerGroup = MTL::Size(std::min(gridSize, 1024), 1, 1);

    encoder->dispatchThreads(threadsPerGrid, threadsPerGroup);
    
    optimizerWeightsQ_->encode(encoder, bufferQ_, inputDim_ * modelDim_, batchSize);
    optimizerWeightsK_->encode(encoder, bufferK_, inputDim_ * modelDim_, batchSize);
    optimizerWeightsV_->encode(encoder, bufferV_, inputDim_ * modelDim_, batchSize);
    optimizerOutputProjection_->encode(encoder, outputProjection_, inputDim_ * modelDim_, batchSize);
    
    encoder->endEncoding();
}

void SelfAttentionLayer::resetErrors() {
    float* errorsBuffer = static_cast<float*>(inputBuffers_[BufferType::IncomingErrors]->contents());
    memset(errorsBuffer, 0, inputBuffers_[BufferType::IncomingErrors]->length());
    inputBuffers_[BufferType::IncomingErrors]->didModifyRange(
        NS::Range::Make(0, inputBuffers_[BufferType::IncomingErrors]->length())
    );
}

void SelfAttentionLayer::setInputBuffer(BufferType type, MTL::Buffer* buffer) {
    inputBuffers_[type] = buffer;
}

MTL::Buffer* SelfAttentionLayer::getOutputBuffer(BufferType type) { return outputBuffers_[type]; }

void SelfAttentionLayer::setOutputBuffer(BufferType type, MTL::Buffer* buffer) {
    outputBuffers_[type] = buffer;
}
MTL::Buffer* SelfAttentionLayer::getInputBuffer(BufferType type) { return inputBuffers_[type]; }


void SelfAttentionLayer::connectForwardConnections(Layer* previousLayer) {
    setInputBuffer(BufferType::Input, previousLayer->getOutputBuffer(BufferType::Output));
}

void SelfAttentionLayer::connectBackwardConnections(Layer* prevLayer)
{
    prevLayer->setInputBuffer(BufferType::IncomingErrors, getOutputBuffer(BufferType::OutgoingErrors));
}

void SelfAttentionLayer::debugLog() {}
void SelfAttentionLayer::onForwardComplete(MTL::CommandQueue*, int) {
    
}

void SelfAttentionLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
}

void SelfAttentionLayer::saveParameters(std::ostream&) const {}
void SelfAttentionLayer::loadParameters(std::istream&) {}

void SelfAttentionLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
