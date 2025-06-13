//
//  dropout-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//
#include <iostream>
#include <random>

#include "logger.h"
#include "input-layer.h"
#include "dropout-layer.h"
#include "training-manager.h"


DropoutLayer::DropoutLayer(float rate, int inputDim, int outputDim, int batchSize, int sequenceLength) :
rate_(rate),
featureDim_(inputDim * batchSize),
inputDim_(inputDim),
batchSize_(batchSize),
sequenceLength_(sequenceLength),
bufferRandomMask_(nullptr),
forwardPipelineState_(nullptr),
backwardPipelineState_(nullptr),
gen(rd()),
dist(0.f, 1.f),
isTerminal_(false) {
    assert(inputDim_ == outputDim);
    
    inputBuffers_[BufferType::Input].resize(1, nullptr);
    outputBuffers_[BufferType::Output].resize(1, nullptr);
    outputBuffers_[BufferType::Debug].resize(1, nullptr);
    
    inputBuffers_[BufferType::IncomingErrors].resize(1, nullptr);
    outputBuffers_[BufferType::OutgoingErrors].resize(1, nullptr);
}

DropoutLayer::~DropoutLayer() {
    for (int t = 0; t < sequenceLength_; ++t) {
        for (auto ob : outputBuffers_) {
            ob.second[t]->release();
        }
    }
    
    if(bufferRandomMask_) bufferRandomMask_->release();
    if(forwardPipelineState_) forwardPipelineState_->release();
    if(backwardPipelineState_) backwardPipelineState_->release();
}

void DropoutLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;
    _pDevice = device;
    
    auto forwardFunction = library->newFunction(NS::String::string("forward_dropout", NS::UTF8StringEncoding));
    forwardPipelineState_ = device->newComputePipelineState(forwardFunction, &error);
    if (!forwardPipelineState_) {
        std::cerr << "Forward pipeline error (Dropout): "
        << error->localizedDescription()->utf8String() << std::endl;
    }
    assert(forwardPipelineState_);
    forwardFunction->release();
    
    auto backwardFunction = library->newFunction(NS::String::string("backward_dropout", NS::UTF8StringEncoding));
    backwardPipelineState_ = device->newComputePipelineState(backwardFunction, &error);
    if (!backwardPipelineState_) {
        std::cerr << "Backward pipeline error (Dropout): "
        << error->localizedDescription()->utf8String() << std::endl;
    }
    assert(backwardPipelineState_);
    backwardFunction->release();
}

void DropoutLayer::buildBuffers(MTL::Device* device) {
    assert(device && "Device is null!");
    
    
    outputBuffers_[BufferType::Output].resize(sequenceLength_);
    inputBuffers_[BufferType::IncomingErrors].resize(sequenceLength_);
    outputBuffers_[BufferType::OutgoingErrors].resize(sequenceLength_);
    outputBuffers_[BufferType::Debug].resize(sequenceLength_);
    
    int t = 0;
    

    outputBuffers_[BufferType::OutgoingErrors][t] = device->newBuffer(featureDim_ * batchSize_ * sizeof(float), MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::Output][t] = device->newBuffer(featureDim_ * batchSize_ * sizeof(float), MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::Debug][t] = device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    
    memset(outputBuffers_[BufferType::Output][t]->contents(), 0, featureDim_ * batchSize_ * sizeof(float));
    memset(outputBuffers_[BufferType::OutgoingErrors][t]->contents(), 0, featureDim_ * batchSize_ * sizeof(float));
    memset(outputBuffers_[BufferType::Debug][t]->contents(), 0, featureDim_ * sizeof(float));
    
    outputBuffers_[BufferType::Output][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Output][t]->length()));
    outputBuffers_[BufferType::OutgoingErrors][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::OutgoingErrors][t]->length()));
    outputBuffers_[BufferType::Debug][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Debug][t]->length()));
    
    Logger::log << "dropout output error buffer initalized @" << outputBuffers_[BufferType::OutgoingErrors][t] << std::endl;
    
    generateRandomMask();
    assert(bufferRandomMask_ && "Random mask buffer allocation failed");
}


void DropoutLayer::forward(MTL::CommandBuffer* cmdBuf, int batchSize) {
    bool isTraining = TrainingManager::instance().isTraining();
    
    if (isTraining) {
        generateRandomMask();
    }
    
    int t = 0;
    
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    encoder->setBuffer(inputBuffers_[BufferType::Input][t], 0, 0);
    encoder->setBuffer(outputBuffers_[BufferType::Output][t], 0, 1);
    encoder->setBuffer(bufferRandomMask_, 0, 2);
    encoder->setBytes(&rate_, sizeof(float), 3);
    encoder->setBytes(&featureDim_, sizeof(int), 4);
    encoder->setBytes(&isTraining, sizeof(bool), 5);
    encoder->setBuffer(outputBuffers_[BufferType::Debug][t], 0, 6);
    
    MTL::Size threadsPerGroup = MTL::Size(std::min(featureDim_, 1024), 1, 1);
    MTL::Size threadgroups = MTL::Size((featureDim_ + 1023) / 1024, 1, 1);
    encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
    encoder->endEncoding();
    
    inputBuffers_[BufferType::Input][t]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Input][t]->length()));

}

void DropoutLayer::backward(MTL::CommandBuffer* cmdBuf, int batchSize) {
    for(int t = 0; t < sequenceLength_; ++t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(backwardPipelineState_);
        encoder->setBuffer(inputBuffers_[BufferType::IncomingErrors][t], 0, 0);
        encoder->setBuffer(outputBuffers_[BufferType::OutgoingErrors][t], 0, 1);
        encoder->setBuffer(bufferRandomMask_, 0, 2);
        encoder->setBytes(&rate_, sizeof(float), 3);
        encoder->setBytes(&featureDim_, sizeof(int), 4);
        encoder->setBuffer(outputBuffers_[BufferType::Debug][t], 0, 5);
        
        MTL::Size threadsPerGroup = MTL::Size(std::min(featureDim_, 1024), 1, 1);
        MTL::Size threadgroups = MTL::Size((featureDim_ + 1023) / 1024, 1, 1);
        encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
        encoder->endEncoding();
    }
}

void DropoutLayer::generateRandomMask() {
    if (!bufferRandomMask_) {
        bufferRandomMask_ = _pDevice->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    }

    float* maskData = (float*)bufferRandomMask_->contents();
    for (int i = 0; i < featureDim_; ++i) {
        maskData[i] = (dist(gen) > rate_) ? 1.0f : 0.0f;
    }

    bufferRandomMask_->didModifyRange(NS::Range(0, bufferRandomMask_->length()));
}

void DropoutLayer::setInputBuffer(BufferType type, MTL::Buffer* buffer) {
    assert(buffer && "Setting input buffer to NULL");
    inputBuffers_[type][0] = buffer;
}

MTL::Buffer* DropoutLayer::getOutputBuffer(BufferType type) {
    return outputBuffers_[type][0];
}

void DropoutLayer::setOutputBuffer(BufferType type, MTL::Buffer* buffer) {
    outputBuffers_[type][0] = buffer;
}

MTL::Buffer* DropoutLayer::getInputBuffer(BufferType type) {
    return inputBuffers_[type][0];
}

void DropoutLayer::resetErrors() {
    float* errorsBuffer = static_cast<float*>(inputBuffers_[BufferType::IncomingErrors][0]->contents());
    memset(errorsBuffer, 0, inputBuffers_[BufferType::IncomingErrors][0]->length());
    inputBuffers_[BufferType::IncomingErrors][0]->didModifyRange(
        NS::Range::Make(0, inputBuffers_[BufferType::IncomingErrors][0]->length())
    );
}

void DropoutLayer::connectForwardConnections(Layer* previousLayer) {
    setInputBuffer(BufferType::Input, previousLayer->getOutputBuffer(BufferType::Output));
}

void DropoutLayer::connectBackwardConnections(Layer* prevLayer)
{
    Logger::log << "dropout output error buffer @" << getOutputBuffer(BufferType::OutgoingErrors) << std::endl;
    prevLayer->setInputBuffer(BufferType::IncomingErrors, getOutputBuffer(BufferType::OutgoingErrors));
}

void DropoutLayer::saveParameters(std::ostream& os) const {
    // No parameters to save
}

void DropoutLayer::loadParameters(std::istream& is) {
    // No parameters to load
}


void DropoutLayer::onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
}

void DropoutLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
}

void DropoutLayer::debugLog() {
    Logger::instance().assertBufferContentsAreValid(outputBuffers_[BufferType::Debug][0], getName() + " debug");
    Logger::instance().assertBufferContentsAreValid(inputBuffers_[BufferType::Input][0], getName() + " input");
    Logger::instance().assertBufferContentsAreValid(outputBuffers_[BufferType::Output][0], getName() + " output");
}
