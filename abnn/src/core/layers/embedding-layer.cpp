//
//  embedding-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include "embedding-layer.h"
#include "logger.h"
#include "adam-optimizer.h"
#include "configuration-manager.h"
#include "weight-initializer.h"


EmbeddingLayer::EmbeddingLayer(int vocabSize, int embeddingDim, int sequenceLength, int outputDim, int batchSize) :
vocabSize_(vocabSize),
embeddingDim_(embeddingDim),
sequenceLength_(sequenceLength),
batchSize_(batchSize),
isTerminal_(false),
forwardPipelineState_(nullptr),
backwardPipelineState_(nullptr),
embeddingsBuffer_(nullptr),
learningRate_(0.001),
initializer_("xavier")
{
    assert(embeddingDim_ == outputDim);
}

EmbeddingLayer::~EmbeddingLayer() {
    if (embeddingsBuffer_) embeddingsBuffer_->release();
    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void EmbeddingLayer::buildBuffers(MTL::Device* device) {
    size_t bufferSize = batchSize_ * embeddingDim_ * sequenceLength_ * sizeof(float);
    
    outputBuffers_[BufferType::Output] = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    
    Logger::log << "embedding output buffer size exp=" << bufferSize << " act=" << outputBuffers_[BufferType::Output]->length() << std::endl;
    
    outputBuffers_[BufferType::OutgoingErrors] = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    
    int embeddingsSize = vocabSize_ * embeddingDim_ * sizeof(float);
    embeddingsBuffer_ = device->newBuffer(embeddingsSize, MTL::ResourceStorageModeManaged);
    float* w = static_cast<float*>(embeddingsBuffer_->contents());
    if (initializer_ == "he") {
        WeightInitializer::initializeHe(w, vocabSize_, embeddingDim_);
    } else {
        WeightInitializer::initializeXavier(w, vocabSize_, embeddingDim_);
    }
    embeddingsBuffer_->didModifyRange(NS::Range(0, embeddingsBuffer_->length()));
    
    optimizerEmbeddings_->buildBuffers(device, embeddingsSize);
}

void EmbeddingLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;
    
    auto forwardFn = library->newFunction(NS::String::string("forward_embedding", NS::UTF8StringEncoding));
    assert(forwardFn && "Forward function not found.");

    auto backwardFn = library->newFunction(NS::String::string("backward_embedding", NS::UTF8StringEncoding));
    assert(backwardFn && "Backward function not found.");

    forwardPipelineState_ = device->newComputePipelineState(forwardFn, &error);
    assert(forwardPipelineState_);

    backwardPipelineState_ = device->newComputePipelineState(backwardFn, &error);
    assert(backwardPipelineState_);
    
    ModelConfig* pConfig = ConfigurationManager::instance().getConfig();
    auto optimizerConfig = pConfig->training.optimizer;
    
    uint accumulation_interval = optimizerConfig.accumulation_interval;
    float learningRate = optimizerConfig.learning_rate;
    float beta1 = optimizerConfig.beta1;
    float beta2 = optimizerConfig.beta2;
    float epsilon = optimizerConfig.epsilon;
    
    optimizerEmbeddings_ = std::make_unique<AdamOptimizer>(learningRate, beta1, beta2, epsilon, accumulation_interval);
    optimizerEmbeddings_->buildPipeline(device, library);
}

void EmbeddingLayer::forward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    encoder->setBuffer(inputBuffers_[BufferType::Input], 0, 0);
    encoder->setBuffer(embeddingsBuffer_, 0, 1);
    encoder->setBuffer(outputBuffers_[BufferType::Output], 0, 2);
    encoder->setBytes(&embeddingDim_, sizeof(int), 3);

    MTL::Size gridSize = MTL::Size(batchSize_ * sequenceLength_, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(std::min(1024, batchSize_ * sequenceLength_), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    
    encoder->endEncoding();
}

void EmbeddingLayer::backward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);

    encoder->setBuffer(outputBuffers_[BufferType::OutgoingErrors], 0, 0);
    encoder->setBuffer(inputBuffers_[BufferType::Input], 0, 1);
    encoder->setBuffer(optimizerEmbeddings_->gradientBuffer(), 0, 2);
    encoder->setBytes(&embeddingDim_, sizeof(uint), 3);

    MTL::Size gridSize = MTL::Size(batchSize_ * sequenceLength_, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(std::min(1024, batchSize_ * sequenceLength_), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    
    optimizerEmbeddings_->encode(encoder, embeddingsBuffer_, embeddingDim_, batchSize);
    
    encoder->endEncoding();
}

void EmbeddingLayer::setInputBuffer(BufferType type, MTL::Buffer* buffer) {
    inputBuffers_[type] = buffer;
}

MTL::Buffer* EmbeddingLayer::getOutputBuffer(BufferType type) { return outputBuffers_[type]; }
void EmbeddingLayer::setOutputBuffer(BufferType type, MTL::Buffer* buffer) {
    outputBuffers_[type] = buffer;
}
MTL::Buffer* EmbeddingLayer::getInputBuffer(BufferType type) { return inputBuffers_[type]; }

int EmbeddingLayer::inputSize() const { return embeddingDim_; }
int EmbeddingLayer::outputSize() const { return embeddingDim_; }

void EmbeddingLayer::updateTargetBufferAt(const float* targetData) {
    assert(false && "EmbeddingLayer cannot be used as a terminal layer with targets.");
}

void EmbeddingLayer::updateTargetBufferAt(const float* targetData, int batchSize) {
    assert(false && "EmbeddingLayer cannot be used as a terminal layer with targets.");
}

void EmbeddingLayer::connectForwardConnections(Layer* previousLayer) {
    setInputBuffer(BufferType::Input, previousLayer->getOutputBuffer(BufferType::Output));
}

void EmbeddingLayer::connectBackwardConnections(Layer* prevLayer)
{
    prevLayer->setInputBuffer(BufferType::IncomingErrors, getOutputBuffer(BufferType::OutgoingErrors));
}

void EmbeddingLayer::resetErrors() {
    float* errorsBuffer = static_cast<float*>(inputBuffers_[BufferType::IncomingErrors]->contents());
    memset(errorsBuffer, 0, inputBuffers_[BufferType::IncomingErrors]->length());
    inputBuffers_[BufferType::IncomingErrors]->didModifyRange(
        NS::Range::Make(0, inputBuffers_[BufferType::IncomingErrors]->length())
    );
}

void EmbeddingLayer::debugLog() {
    //Logger::instance().printFloatBuffer(outputBuffers_[BufferType::Output], getName() + " D output", 100);
    Logger::instance().assertBufferContentsAreValid(outputBuffers_[BufferType::Output], getName() + " D output");
    
    /*
    Logger::instance().assertBufferContentsAreValid(inputBuffers_[BufferType::Targets], getName() + " D targets");
    Logger::instance().assertBufferContentsAreValid(embeddingsBuffer_, getName() + " D embeddings");
    Logger::instance().assertBufferContentsAreValid(optimizerEmbeddings_->gradientBuffer(), getName() + " D embeddings gradients");
    Logger::instance().assertBufferContentsAreValid(inputBuffers_[BufferType::Input], getName() + " D input");
    Logger::instance().assertBufferContentsAreValid(outputBuffers_[BufferType::Output], getName() + " D output");
    */
    //Logger::instance().printFloatBuffer(outputBuffers_[BufferType::Output], "D Embedding output");
}

void EmbeddingLayer::onForwardComplete(MTL::CommandQueue*, int) {
}

void EmbeddingLayer::onBackwardComplete(MTL::CommandQueue*, int) {
}

void EmbeddingLayer::saveParameters(std::ostream&) const {}
void EmbeddingLayer::loadParameters(std::istream&) {}

void EmbeddingLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
