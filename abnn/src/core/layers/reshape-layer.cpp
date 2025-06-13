//
//  rehape-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include "reshape-layer.h"
#include "logger.h"

ReshapeLayer::ReshapeLayer(int sequenceLength, int inputSize, int outputSize, int batchSize) :
    sequenceLength_(sequenceLength),
    inputSize_(inputSize),
    outputSize_(outputSize),
    batchSize_(batchSize),
    isTerminal_(false),
    forwardPipelineState_(nullptr),
    backwardPipelineState_(nullptr) {
        assert(inputSize_ == sequenceLength_ * outputSize_ &&
           "ReshapeLayer dimension mismatch: inputSize must equal sequenceLength * outputSize");
}

ReshapeLayer::~ReshapeLayer() {}

void ReshapeLayer::buildBuffers(MTL::Device* device) {
    // Explicitly no buffer allocation required, reusing buffers from connected layers
}

void ReshapeLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    // Explicitly no compute pipeline needed for ReshapeLayer
}

void ReshapeLayer::forward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    //Intentionally blank
}

void ReshapeLayer::backward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    //Intentionally blank
}

void ReshapeLayer::resetErrors() {
    //Intentionally blank
}

void ReshapeLayer::setInputBuffer(BufferType type, MTL::Buffer* buffer) {
    inputBuffers_[type] = buffer;
}

MTL::Buffer* ReshapeLayer::getOutputBuffer(BufferType type) { return outputBuffers_[type]; }
void ReshapeLayer::setOutputBuffer(BufferType type, MTL::Buffer* buffer) {
    outputBuffers_[type] = buffer;
}
MTL::Buffer* ReshapeLayer::getInputBuffer(BufferType type) { return inputBuffers_[type]; }

int ReshapeLayer::inputSize() const { return inputSize_; }
int ReshapeLayer::outputSize() const { return outputSize_; }

void ReshapeLayer::updateTargetBufferAt(const float* targetData) {
    assert(false && "ReshapeLayer cannot be used as a terminal layer with targets.");
}

void ReshapeLayer::updateTargetBufferAt(const float* targetData, int batchSize) {
    assert(false && "ReshapeLayer cannot be used as a terminal layer with targets.");
}

void ReshapeLayer::connectForwardConnections(Layer* previousLayer) {
    setInputBuffer(BufferType::Input, previousLayer->getOutputBuffer(BufferType::Output));
    setOutputBuffer(BufferType::Output, this->getInputBuffer(BufferType::Input));
}

void ReshapeLayer::connectBackwardConnections(Layer* prevLayer)
{
    setOutputBuffer(BufferType::OutgoingErrors, this->getInputBuffer(BufferType::IncomingErrors));
    prevLayer->setInputBuffer(BufferType::IncomingErrors, getOutputBuffer(BufferType::OutgoingErrors));
}

void ReshapeLayer::debugLog() {}

void ReshapeLayer::onForwardComplete(MTL::CommandQueue*, int) {
}

void ReshapeLayer::onBackwardComplete(MTL::CommandQueue*, int) {
}

void ReshapeLayer::saveParameters(std::ostream&) const {}
void ReshapeLayer::loadParameters(std::istream&) {}

void ReshapeLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
