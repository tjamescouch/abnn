//
//  residual-connection-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include "residual-connection-layer.h"
#include "logger.h"

ResidualConnectionLayer::ResidualConnectionLayer(int featureDim, int sequenceLength, int batchSize, float residualScale) :
featureDim_(featureDim),
sequenceLength_(sequenceLength),
batchSize_(batchSize),
residualScale_(residualScale),
isTerminal_(false),
fromLayer_(nullptr),
forwardPipelineState_(nullptr),
backwardPipelineState_(nullptr) {}

ResidualConnectionLayer::~ResidualConnectionLayer() {}

ResidualConnectionLayer* ResidualConnectionLayer::setFromLayer(Layer* fromLayer) {
    fromLayer_ = fromLayer;
    
    return this;
}

void ResidualConnectionLayer::buildBuffers(MTL::Device* device) {
    size_t bufferSize = batchSize_ * featureDim_ * sequenceLength_ * sizeof(float);
    
    outputBuffers_[BufferType::Output] = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::OutgoingErrors] = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
}

void ResidualConnectionLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;
    
    auto forwardFn = library->newFunction(NS::String::string("forward_residual", NS::UTF8StringEncoding));
    assert(forwardFn && "Forward function not found.");

    auto backwardFn = library->newFunction(NS::String::string("backward_residual", NS::UTF8StringEncoding));
    assert(backwardFn && "Backward function not found.");

    forwardPipelineState_ = device->newComputePipelineState(forwardFn, &error);
    assert(forwardPipelineState_);

    backwardPipelineState_ = device->newComputePipelineState(backwardFn, &error);
    assert(backwardPipelineState_);
}

void ResidualConnectionLayer::forward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    encoder->setBuffer(inputBuffers_[BufferType::Input], 0, 0);
    encoder->setBuffer(fromLayer_->getOutputBuffer(BufferType::Output), 0, 1);
    encoder->setBuffer(outputBuffers_[BufferType::Output], 0, 2);
    encoder->setBytes(&residualScale_, sizeof(float), 3);

    MTL::Size gridSize = MTL::Size(batchSize_ * sequenceLength_ * featureDim_, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(std::min(1024, batchSize_ * sequenceLength_ * featureDim_), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
}

void ResidualConnectionLayer::backward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);

    encoder->setBuffer(inputBuffers_[BufferType::IncomingErrors], 0, 0);   // input error coming from next layer
    encoder->setBuffer(outputBuffers_[BufferType::OutgoingErrors], 0, 1); // propagate back to previous layer
    encoder->setBuffer(fromLayer_->getInputBuffer(BufferType::IncomingErrors), 0, 2);               // propagate error back to residual source layer
    encoder->setBytes(&residualScale_, sizeof(float), 3);

    MTL::Size gridSize = MTL::Size(batchSize_ * sequenceLength_ * featureDim_, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(std::min(1024, batchSize_ * sequenceLength_ * featureDim_), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
}

void ResidualConnectionLayer::setInputBuffer(BufferType type, MTL::Buffer* buffer) {
    inputBuffers_[type] = buffer;
}

MTL::Buffer* ResidualConnectionLayer::getOutputBuffer(BufferType type) { return outputBuffers_[type]; }
void ResidualConnectionLayer::setOutputBuffer(BufferType type, MTL::Buffer* buffer) {
    outputBuffers_[type] = buffer;
}
MTL::Buffer* ResidualConnectionLayer::getInputBuffer(BufferType type) { return inputBuffers_[type]; }

int ResidualConnectionLayer::inputSize() const { return featureDim_; }
int ResidualConnectionLayer::outputSize() const { return featureDim_; }

void ResidualConnectionLayer::updateTargetBufferAt(const float* targetData) {
    assert(false && "ResidualConnectionLayer cannot be used as a terminal layer with targets.");
}

void ResidualConnectionLayer::updateTargetBufferAt(const float* targetData, int batchSize) {
    assert(false && "ResidualConnectionLayer cannot be used as a terminal layer with targets.");
}

void ResidualConnectionLayer::connectForwardConnections(Layer* previousLayer) {
    setInputBuffer(BufferType::Input, previousLayer->getOutputBuffer(BufferType::Output));
}

void ResidualConnectionLayer::connectBackwardConnections(Layer* prevLayer)
{
    prevLayer->setInputBuffer(BufferType::IncomingErrors, getOutputBuffer(BufferType::OutgoingErrors));
}

void ResidualConnectionLayer::resetErrors() {
    float* errorsBuffer = static_cast<float*>(inputBuffers_[BufferType::IncomingErrors]->contents());
    memset(errorsBuffer, 0, inputBuffers_[BufferType::IncomingErrors]->length());
    inputBuffers_[BufferType::IncomingErrors]->didModifyRange(
        NS::Range::Make(0, inputBuffers_[BufferType::IncomingErrors]->length())
    );
}

void ResidualConnectionLayer::debugLog() {}

void ResidualConnectionLayer::onForwardComplete(MTL::CommandQueue*, int) {
    Logger::instance().assertBufferContentsAreValid(outputBuffers_[BufferType::Output], getName());
}

void ResidualConnectionLayer::onBackwardComplete(MTL::CommandQueue*, int) {
    Logger::instance().assertBufferContentsAreValid(outputBuffers_[BufferType::Output], getName());
}

void ResidualConnectionLayer::saveParameters(std::ostream&) const {}
void ResidualConnectionLayer::loadParameters(std::istream&) {}

void ResidualConnectionLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
