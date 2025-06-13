//
//  map-reduce-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-04.
//
#include "map-reduce-layer.h"
#include <stdexcept>
#include <iostream>
#include "logger.h"


MapReduceLayer::MapReduceLayer(int inputSize, int outputSize, ReductionType reductionType)
: inputSize_(inputSize), output_dim_(outputSize),
reductionType_(reductionType),
forwardPipelineState_(nullptr),
backwardPipelineState_(nullptr),
isTerminal_(false) {
    assert(outputSize == 1);
    inputBuffers_[BufferType::IncomingErrors].resize(1, nullptr);
    inputBuffers_[BufferType::Input].resize(1, nullptr);
    outputBuffers_[BufferType::Output].resize(1, nullptr);
    outputBuffers_[BufferType::Delta].resize(1, nullptr);
    outputBuffers_[BufferType::OutgoingErrors].resize(1, nullptr);
}


MapReduceLayer::~MapReduceLayer() {
    for (auto ob : outputBuffers_) {
        ob.second[0]->release();
    }
    
    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void MapReduceLayer::buildBuffers(MTL::Device* device) {
    inputBuffers_[BufferType::IncomingErrors].clear();
    inputBuffers_[BufferType::Input].clear();
    outputBuffers_[BufferType::Output].clear();
    outputBuffers_[BufferType::Delta].clear();
    outputBuffers_[BufferType::OutgoingErrors].clear();
    
    inputBuffers_[BufferType::Input].push_back(device->newBuffer(inputSize_ * sizeof(float), MTL::ResourceStorageModeManaged));
    inputBuffers_[BufferType::IncomingErrors].push_back(device->newBuffer(inputSize_ * sizeof(float), MTL::ResourceStorageModeManaged));
    outputBuffers_[BufferType::Output].push_back(device->newBuffer(output_dim_ * sizeof(float), MTL::ResourceStorageModeManaged));
    outputBuffers_[BufferType::Delta].push_back(device->newBuffer(output_dim_ * sizeof(float), MTL::ResourceStorageModeManaged));
    outputBuffers_[BufferType::OutgoingErrors].push_back(device->newBuffer(output_dim_ * sizeof(float), MTL::ResourceStorageModeManaged));
}

void MapReduceLayer::resetErrors() {
    float* errorsBuffer = static_cast<float*>(inputBuffers_[BufferType::IncomingErrors][0]->contents());
    memset(errorsBuffer, 0, inputBuffers_[BufferType::IncomingErrors][0]->length());
    inputBuffers_[BufferType::IncomingErrors][0]->didModifyRange(
        NS::Range::Make(0, inputBuffers_[BufferType::IncomingErrors][0]->length())
    );
}

void MapReduceLayer::connectForwardConnections(Layer* previousLayer){
    setInputBuffer(BufferType::Input, previousLayer->getOutputBuffer(BufferType::Output));
}

void MapReduceLayer::connectBackwardConnections(Layer* prevLayer)
{
    prevLayer->setInputBuffer(BufferType::IncomingErrors, getOutputBuffer(BufferType::OutgoingErrors));
}

void MapReduceLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    auto kernelNameForward = NS::String::string("forward_map_reduce", NS::UTF8StringEncoding);
    auto kernelNameBackward = NS::String::string("backward_map_reduce", NS::UTF8StringEncoding);
    
    MTL::Function* forwardFunction = library->newFunction(kernelNameForward);
    MTL::Function* backwardFunction = library->newFunction(kernelNameBackward);
    
    NS::Error* error = nullptr;
    
    forwardPipelineState_ = device->newComputePipelineState(forwardFunction, &error);
    if (!forwardPipelineState_) {
        throw std::runtime_error("Error creating forward pipeline state");
    }
    
    backwardPipelineState_ = device->newComputePipelineState(backwardFunction, &error);
    if (!backwardPipelineState_) {
        throw std::runtime_error("Error creating backward pipeline state");
    }
    
    forwardFunction->release();
    backwardFunction->release();
}

void MapReduceLayer::forward(MTL::CommandBuffer* cmdBuf, int batchSize) {
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    
    encoder->setBuffer(inputBuffers_[BufferType::Input][0], 0, 0);
    encoder->setBuffer(outputBuffers_[BufferType::Output][0], 0, 1);
    encoder->setBytes(&inputSize_, sizeof(int), 2);
    encoder->setBytes(&reductionType_, sizeof(int), 3);
    
    MTL::Size gridSize = MTL::Size(inputSize_, 1, 1);
    MTL::Size threadgroupSize = MTL::Size(std::min(inputSize_, 1024), 1, 1);
    encoder->dispatchThreads(gridSize, threadgroupSize);
    
    encoder->endEncoding();
}

void MapReduceLayer::backward(MTL::CommandBuffer* cmdBuf, int batchSize) {
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);
    
    encoder->setBuffer(outputBuffers_[BufferType::Delta][0], 0, 0);
    encoder->setBuffer(outputBuffers_[BufferType::Output][0], 0, 1); // forwardOutput buffer
    encoder->setBuffer(inputBuffers_[BufferType::IncomingErrors][0], 0, 2);
    encoder->setBytes(&inputSize_, sizeof(uint), 3);
    encoder->setBytes(&reductionType_, sizeof(uint), 4);
    
    MTL::Size gridSize = MTL::Size(inputSize_, 1, 1);
    MTL::Size threadgroupSize = MTL::Size(fmin(inputSize_, 1024u), 1, 1);
    encoder->dispatchThreads(gridSize, threadgroupSize);
    
    encoder->endEncoding();
}


void MapReduceLayer::setInputBuffer(BufferType type, MTL::Buffer* buffer) {
    inputBuffers_[type][0] = buffer;
}

MTL::Buffer* MapReduceLayer::getOutputBuffer(BufferType type) {
    return outputBuffers_[type][0];
}

void MapReduceLayer::setOutputBuffer(BufferType type, MTL::Buffer* buffer) {
    outputBuffers_[type][0] = buffer;
}

MTL::Buffer* MapReduceLayer::getInputBuffer(BufferType type) {
    return inputBuffers_[type][0];
}

int MapReduceLayer::inputSize() const {
    return inputSize_;
}

int MapReduceLayer::outputSize() const {
    return output_dim_;
}

void MapReduceLayer::updateTargetBufferAt(const float* targetData) {
    // Typically a MapReduceLayer might ignore this or handle it differently
}

void MapReduceLayer::updateTargetBufferAt(const float* targetData, int batchSize) {
    // Typically a MapReduceLayer might ignore this or handle it differently
}

void MapReduceLayer::debugLog() {
    Logger::log << "[MapReduceLayer] debugLog called." << std::endl;
}

void MapReduceLayer::onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
    // Implement if necessary
}

void MapReduceLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
}

void MapReduceLayer::saveParameters(std::ostream& outStream) const {
    // Implement parameter serialization
}

void MapReduceLayer::loadParameters(std::istream& inStream) {
    // Implement parameter deserialization
}

void MapReduceLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
