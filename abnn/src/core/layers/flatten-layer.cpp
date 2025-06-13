//
//  reshape-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include "flatten-layer.h"
#include "logger.h"

FlattenLayer::FlattenLayer(int sequenceLength, int inputSize, int outputSize, int batchSize) :
    sequenceLength_(sequenceLength),
    inputSize_(inputSize),
    outputSize_(outputSize),
    batchSize_(batchSize),
    isTerminal_(false),
    forwardPipelineState_(nullptr),
    backwardPipelineState_(nullptr) {
    assert(outputSize_ == sequenceLength_ * inputSize &&
           "FlattenLayer output size mismatch: outputSize must equal sequenceLength * inputSize");
}

FlattenLayer::~FlattenLayer() {}

void FlattenLayer::buildBuffers(MTL::Device* device) {
    // Explicitly no buffer allocation required, reusing buffers from connected layers
}

void FlattenLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    // Explicitly no compute pipeline needed for FlattenLayer
}

void FlattenLayer::forward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    //Intentionally empty
}

void FlattenLayer::backward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    //Intentionally empty
}

void FlattenLayer::setInputBuffer(BufferType type, MTL::Buffer* buffer) {
    inputBuffers_[type] = buffer;
}

MTL::Buffer* FlattenLayer::getOutputBuffer(BufferType type) { return outputBuffers_[type]; }
void FlattenLayer::setOutputBuffer(BufferType type, MTL::Buffer* buffer) {
    outputBuffers_[type] = buffer;
}
MTL::Buffer* FlattenLayer::getInputBuffer(BufferType type) { return inputBuffers_[type]; }

int FlattenLayer::inputSize() const { return inputSize_; }
int FlattenLayer::outputSize() const { return outputSize_; }

void FlattenLayer::resetErrors() {
    //Intentionally blank
}

void FlattenLayer::updateTargetBufferAt(const float* targetData) {
    assert(false && "FlattenLayer cannot be used as a terminal layer with targets.");
}

void FlattenLayer::updateTargetBufferAt(const float* targetData, int batchSize) {
    assert(false && "FlattenLayer cannot be used as a terminal layer with targets.");
}

void FlattenLayer::connectForwardConnections(Layer* previousLayer) {
    setInputBuffer(BufferType::Input, previousLayer->getOutputBuffer(BufferType::Output));
    setOutputBuffer(BufferType::Output, this->getInputBuffer(BufferType::Input));
}

void FlattenLayer::connectBackwardConnections(Layer* prevLayer)
{
    setOutputBuffer(BufferType::OutgoingErrors, this->getInputBuffer(BufferType::IncomingErrors));
    prevLayer->setInputBuffer(BufferType::IncomingErrors, getOutputBuffer(BufferType::OutgoingErrors));
}

void FlattenLayer::debugLog() {}

void FlattenLayer::onForwardComplete(MTL::CommandQueue*, int) {
}

void FlattenLayer::onBackwardComplete(MTL::CommandQueue*, int) {
}

void FlattenLayer::saveParameters(std::ostream&) const {}
void FlattenLayer::loadParameters(std::istream&) {}

void FlattenLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
