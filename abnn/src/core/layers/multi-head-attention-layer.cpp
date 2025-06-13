//
//  multi-head-attention-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include "multi-head-attention-layer.h"
#include "logger.h"
#include "configuration-manager.h"
#include "model-config.h"
#include "adam-optimizer.h"
#include "weight-initializer.h"

MultiHeadAttentionLayer::MultiHeadAttentionLayer(uint inputDim, uint modelDim, uint seqLength, uint batchSize, uint numHeads) :
inputDim_(inputDim),
modelDim_(modelDim),
seqLength_(seqLength),
batchSize_(batchSize),
numHeads_(numHeads),
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
    assert(modelDim_ % numHeads_ == 0);
    uint headDim = modelDim / numHeads;
    scale_ = 1.0f/sqrt(float(headDim));
}

MultiHeadAttentionLayer::~MultiHeadAttentionLayer() {
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

void MultiHeadAttentionLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;

    auto kernelFn = library->newFunction(NS::String::string("forward_multi_head_attention", NS::ASCIIStringEncoding));
    forwardPipelineState_ = device->newComputePipelineState(kernelFn, &error);
    kernelFn->release();

    if (!forwardPipelineState_) {
        Logger::log << "Error occurred creating forward pipeline: " << error->localizedDescription()->utf8String() << std::endl;
        assert(false);
    }

    auto kernelBackwardFn = library->newFunction(NS::String::string("backward_multi_head_attention", NS::ASCIIStringEncoding));
    backwardPipelineState_ = device->newComputePipelineState(kernelBackwardFn, &error);
    kernelBackwardFn->release();

    if (!backwardPipelineState_) {
        Logger::log << "Error occurred creating backward pipeline: " << error->localizedDescription()->utf8String() << std::endl;
        std::exit(-1);
    }
    
    ModelConfig* pConfig = ConfigurationManager::instance().getConfig();
    auto parameters = pConfig->training.optimizer.parameters;
    auto optimizerConfig = pConfig->training.optimizer;
    float lr = pConfig->training.optimizer.learning_rate; //FIXME get layer specific learning rate in factory
    
    uint accumulation_interval = optimizerConfig.accumulation_interval;
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

void MultiHeadAttentionLayer::buildBuffers(MTL::Device* device) {
    const size_t attentionBufferSize = batchSize_ * seqLength_ * seqLength_ * sizeof(float);
    const size_t activationBufferSize = batchSize_ * seqLength_ * modelDim_ * sizeof(float);
    const size_t errorBufferSize = batchSize_ * seqLength_ * inputDim_ * sizeof(float);
    const size_t weightsBufferSize = inputDim_ * modelDim_ * sizeof(float);
    
    const size_t headDim = modelDim_ / numHeads_;
    
    
    const size_t scratchPerThread=2*(inputDim_)+3*(headDim)+2*(seqLength_ * headDim)+2*(seqLength_);
    const size_t scratchBufferSize=(batchSize_ * seqLength_) * scratchPerThread * sizeof(float);
    
    
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
    outputBuffers_[BufferType::OutgoingErrors] = device->newBuffer(errorBufferSize, MTL::ResourceStorageModeManaged);
    
    outputBuffers_[BufferType::Debug] = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);
    
    optimizerWeightsQ_->buildBuffers(device, weightsBufferSize);
    optimizerWeightsK_->buildBuffers(device, weightsBufferSize);
    optimizerWeightsV_->buildBuffers(device, weightsBufferSize);
    optimizerOutputProjection_->buildBuffers(device, weightsBufferSize);
}

void MultiHeadAttentionLayer::forward(MTL::CommandBuffer* commandBuffer, int batchSize) {
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
    encoder->setBytes(&numHeads_, sizeof(uint), 13);
    encoder->setBytes(&scale_, sizeof(float), 14);
    

    // 6) Calculate thread count: each thread handles exactly 1 token (batch × sequence length)
    const uint32_t totalThreads = batchSize_ * seqLength_;

    // 7) Choose a threadgroup size (e.g., 64)
    const uint32_t threadsPerGroup = 64;

    // 8) Calculate number of threadgroups needed (rounded up)
    uint32_t numThreadgroups = (totalThreads + threadsPerGroup - 1) / threadsPerGroup;

    // 9) Define threadgroup and grid sizes explicitly for dispatch
    MTL::Size threadsPerThreadgroup = MTL::Size::Make(threadsPerGroup, 1, 1);
    MTL::Size threadgroups = MTL::Size::Make(numThreadgroups, 1, 1);

    // 10) Dispatch threadgroups explicitly (consistent with GPU indexing expectations)
    encoder->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);

    
    
    encoder->endEncoding();
}



void MultiHeadAttentionLayer::backward(MTL::CommandBuffer* commandBuffer, int batchSize) {
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
    encoder->setBuffer(optimizerOutputProjection_->gradientBuffer(), 0, 14);      // Gradients for outputProjection

    // Constant arguments (dimensions)
    encoder->setBytes(&bs, sizeof(uint), 15);
    encoder->setBytes(&seqLength_, sizeof(uint), 16);
    encoder->setBytes(&inputDim_, sizeof(uint), 17);
    encoder->setBytes(&modelDim_, sizeof(uint), 18);
    
    //Scratch
    encoder->setBuffer(bufferScratch_, 0, 19); 
    
    encoder->setBytes(&numHeads_, sizeof(uint), 20);
    encoder->setBytes(&scale_, sizeof(float), 21);
    

    // 1) Each thread in this kernel handles exactly one (batchIndex, seqIndex).
    //    So total number of threads needed = batchSize * seqLength.
    const uint32_t totalTokens = batchSize * seqLength_;

    // 2) Pick a threadgroup (aka "block") size, e.g. 256 or 128.
    //    There's no universal best; you usually experiment or pick a typical GPU-friendly size.
    const uint32_t threadsPerGroup = 256;

    // 3) Compute how many threadgroups we need:
    //    Each threadgroup has 'threadsPerGroup' threads,
    //    so numberOfThreadgroups = ceil(totalTokens / threadsPerGroup).
    uint32_t numberOfThreadgroups = (totalTokens + threadsPerGroup - 1) / threadsPerGroup;

    // 4) In Metal, we can dispatch in 1D. We form an MTL::Size for threadgroups,
    //    and one for the threadsPerGroup.
    MTL::Size threadgroupCount = MTL::Size::Make(numberOfThreadgroups, 1, 1);
    MTL::Size threadgroupSize  = MTL::Size::Make(threadsPerGroup, 1, 1);

    // 5) Encode the dispatch:
    //    "dispatchThreadgroups" uses the # of threadgroups and the threadsPerGroup size.
    //    The kernel will see:
    //      blockId in [0..numberOfThreadgroups-1]
    //      tid in [0..threadsPerGroup-1]
    //    Then 'gid = blockId*threadsPerGroup + tid'
    //    (that matches how your kernel calculates its global ID).
    encoder->dispatchThreadgroups(threadgroupCount, threadgroupSize);
    
    optimizerWeightsQ_->encode(encoder, bufferQ_, inputDim_ * modelDim_, batchSize);
    optimizerWeightsK_->encode(encoder, bufferK_, inputDim_ * modelDim_, batchSize);
    optimizerWeightsV_->encode(encoder, bufferV_, inputDim_ * modelDim_, batchSize);
    optimizerOutputProjection_->encode(encoder, outputProjection_, inputDim_ * modelDim_, batchSize);
    
    encoder->endEncoding();
}

void MultiHeadAttentionLayer::setInputBuffer(BufferType type, MTL::Buffer* buffer) {
    inputBuffers_[type] = buffer;
}

MTL::Buffer* MultiHeadAttentionLayer::getOutputBuffer(BufferType type) { return outputBuffers_[type]; }

void MultiHeadAttentionLayer::setOutputBuffer(BufferType type, MTL::Buffer* buffer) {
    outputBuffers_[type] = buffer;
}

MTL::Buffer* MultiHeadAttentionLayer::getInputBuffer(BufferType type) { return inputBuffers_[type]; }

void MultiHeadAttentionLayer::resetErrors() {
    float* errorsBuffer = static_cast<float*>(inputBuffers_[BufferType::IncomingErrors]->contents());
    memset(errorsBuffer, 0, inputBuffers_[BufferType::IncomingErrors]->length());
    inputBuffers_[BufferType::IncomingErrors]->didModifyRange(
        NS::Range::Make(0, inputBuffers_[BufferType::IncomingErrors]->length())
    );
}


void MultiHeadAttentionLayer::connectForwardConnections(Layer* previousLayer) {
    setInputBuffer(BufferType::Input, previousLayer->getOutputBuffer(BufferType::Output));
}

void MultiHeadAttentionLayer::connectBackwardConnections(Layer* prevLayer)
{
    prevLayer->setInputBuffer(BufferType::IncomingErrors, getOutputBuffer(BufferType::OutgoingErrors));
}

void MultiHeadAttentionLayer::debugLog() {
    Logger::instance().assertBufferContentsAreValid(optimizerWeightsQ_->gradientBuffer(), getName() + " D grad optimizerWeightsQ_");
    Logger::instance().assertBufferContentsAreValid(optimizerWeightsK_->gradientBuffer(), getName() + " D grad optimizerWeightsK_");
    Logger::instance().assertBufferContentsAreValid(optimizerWeightsV_->gradientBuffer(), getName() + " D grad optimizerWeightsV_");
    Logger::instance().assertBufferContentsAreValid(optimizerOutputProjection_->gradientBuffer(), getName() + " D grad optimizerOutputProjection_");
    
    Logger::instance().assertBufferContentsAreValid(weightsQ_, getName() + " D weightsQ_");
    Logger::instance().assertBufferContentsAreValid(weightsK_, getName() + " D weightsK_");
    Logger::instance().assertBufferContentsAreValid(weightsV_, getName() + " D weightsV_");
    
    Logger::instance().assertBufferContentsAreValid(outputProjection_, getName() + " D outputProjection_");
    
    
    Logger::instance().assertBufferContentsAreValid(bufferK_, getName() + " D bufferK_");
    Logger::instance().assertBufferContentsAreValid(bufferV_, getName() + " D bufferV_");
    
    Logger::instance().assertBufferContentsAreValid(bufferQ_, getName() + " D bufferQ_");
    
    Logger::instance().assertBufferContentsAreValid(inputBuffers_[BufferType::Input], getName() + " input");
    Logger::instance().assertBufferContentsAreValid(outputBuffers_[BufferType::Output], getName() + " output");
}

void MultiHeadAttentionLayer::onForwardComplete(MTL::CommandQueue*, int) {
    Logger::instance().assertBufferContentsAreValid(outputBuffers_[BufferType::Output], getName() + " F output");
}

void MultiHeadAttentionLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
    Logger::instance().assertBufferContentsAreValid(outputBuffers_[BufferType::Output], getName() + " B output");
}

void MultiHeadAttentionLayer::saveParameters(std::ostream&) const {}
void MultiHeadAttentionLayer::loadParameters(std::istream&) {}

void MultiHeadAttentionLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
