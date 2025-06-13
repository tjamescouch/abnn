//
//  positional-encoding-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include "positional-encoding-layer.h"
#include "logger.h"

PositionalEncodingLayer::PositionalEncodingLayer(int embeddingDim, int sequenceLength, int outputDim, int batchSize) :
    sequenceLength_(sequenceLength),
    embeddingDim_(embeddingDim),
    batchSize_(batchSize),
    isTerminal_(false),
    forwardPipelineState_(nullptr)
{
    assert(embeddingDim_ == outputDim);
}

PositionalEncodingLayer::~PositionalEncodingLayer() {}

void PositionalEncodingLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    auto forwardFunc = library->newFunction(NS::String::string("forward_positional_encoding", NS::UTF8StringEncoding));
    assert(forwardFunc && "Forward function not found.");
    
    NS::Error* error = nullptr;
    forwardPipelineState_ = device->newComputePipelineState(forwardFunc, &error);
    assert(forwardPipelineState_);
    
    forwardFunc->release();
}

void PositionalEncodingLayer::buildBuffers(MTL::Device* device) {
    size_t bufferSize = embeddingDim_  * sequenceLength_ * sizeof(float);
    
    positionalEncodingBuffer_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    
    std::vector<float> positionalEncodings;
    positionalEncodings.resize(sequenceLength_ * embeddingDim_);
    for (int pos = 0; pos < sequenceLength_; ++pos) {
        for (int i = 0; i < embeddingDim_; ++i) {
            float angle = pos / powf(10000.0f, (2 * (i / 2)) / static_cast<float>(embeddingDim_));
            positionalEncodings[pos * embeddingDim_ + i] = (i % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }
}


void PositionalEncodingLayer::forward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    encoder->setBuffer(outputBuffers_[BufferType::Output], 0, 0);
    encoder->setBuffer(positionalEncodingBuffer_, 0, 1);
    encoder->setBytes(&sequenceLength_, sizeof(uint), 2);
    encoder->setBytes(&embeddingDim_, sizeof(uint), 3);

    MTL::Size gridSize = MTL::Size(batchSize_ * sequenceLength_, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(std::min(1024u, batchSize_ * sequenceLength_), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    
    encoder->endEncoding();
}

void PositionalEncodingLayer::backward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    //Intentionally blank
}

void PositionalEncodingLayer::resetErrors() {
    //Intentionally blank
}

void PositionalEncodingLayer::setInputBuffer(BufferType type, MTL::Buffer* buffer) {
    inputBuffers_[type] = buffer;
}

MTL::Buffer* PositionalEncodingLayer::getOutputBuffer(BufferType type) { return outputBuffers_[type]; }
void PositionalEncodingLayer::setOutputBuffer(BufferType type, MTL::Buffer* buffer) {
    outputBuffers_[type] = buffer;
}
MTL::Buffer* PositionalEncodingLayer::getInputBuffer(BufferType type) { return inputBuffers_[type]; }

int PositionalEncodingLayer::inputSize() const { return embeddingDim_; }
int PositionalEncodingLayer::outputSize() const { return embeddingDim_; }

void PositionalEncodingLayer::updateTargetBufferAt(const float* targetData) {
    assert(false && "PositionalEncodingLayer cannot be used as a terminal layer with targets.");
}

void PositionalEncodingLayer::updateTargetBufferAt(const float* targetData, int batchSize) {
    assert(false && "PositionalEncodingLayer cannot be used as a terminal layer with targets.");
}

void PositionalEncodingLayer::connectForwardConnections(Layer* previousLayer) {
    setInputBuffer(BufferType::Input, previousLayer->getOutputBuffer(BufferType::Output));
    setOutputBuffer(BufferType::Output, this->getInputBuffer(BufferType::Input));
}

void PositionalEncodingLayer::connectBackwardConnections(Layer* prevLayer)
{
    setOutputBuffer(BufferType::OutgoingErrors, this->getInputBuffer(BufferType::IncomingErrors));
    prevLayer->setInputBuffer(BufferType::IncomingErrors, getOutputBuffer(BufferType::OutgoingErrors));
}

void PositionalEncodingLayer::debugLog() {}

void PositionalEncodingLayer::onForwardComplete(MTL::CommandQueue*, int) {
}

void PositionalEncodingLayer::onBackwardComplete(MTL::CommandQueue*, int) {
}

void PositionalEncodingLayer::saveParameters(std::ostream&) const {}
void PositionalEncodingLayer::loadParameters(std::istream&) {}

void PositionalEncodingLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
