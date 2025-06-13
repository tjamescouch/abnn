//
//  batch-normalization-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#include "input-layer.h"
#include "adam-optimizer.h"
#include "layer-normalization-layer.h"
#include <cassert>
#include <random>
#include <cstring>
#include <iostream>
#include "training-manager.h"
#include "math-lib.h"
#include "logger.h"
#include "model-config.h"
#include "configuration-manager.h"

LayerNormalizationLayer::LayerNormalizationLayer(int featureDim, int seqLength, int batchSize, float learningRate, float epsilon)
    : featureDim_(featureDim),
      seqLength_(seqLength),
      batchSize_(batchSize),
      learningRate_(learningRate),
      epsilon_(epsilon),
      bufferDebug_(nullptr),
      bufferGamma_(nullptr),
      bufferBeta_(nullptr),
      bufferSavedMean_(nullptr),
      bufferSavedVariance_(nullptr),
      forwardPipelineState_(nullptr),
      backwardPipelineState_(nullptr)
{
}

LayerNormalizationLayer::~LayerNormalizationLayer() {
    if (bufferDebug_) bufferDebug_->release();
    if (bufferGamma_) bufferGamma_->release();
    if (bufferBeta_) bufferBeta_->release();
    
    if (bufferSavedMean_) {
        bufferSavedMean_->release();
        bufferSavedMean_ = nullptr;
    }
    if (bufferSavedVariance_) {
        bufferSavedVariance_->release();
        bufferSavedVariance_ = nullptr;
    }
    
    for (auto ob : outputBuffers_) {
        ob.second[0]->release();
    }
    
    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void LayerNormalizationLayer::initializeParameters(MTL::Device* device) {
    std::vector<float> debug(featureDim_, 0.0f);
    std::vector<float> gamma(featureDim_, 1.0f);
    std::vector<float> beta(featureDim_, 0.0f);

    // This is the size for the per-feature arrays:
    size_t bufferSize = sizeof(float) * featureDim_;

    bufferDebug_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    bufferGamma_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    bufferBeta_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);

    bufferSavedMean_ = device->newBuffer(sizeof(float) * batchSize_ * seqLength_, MTL::ResourceStorageModeManaged);
    bufferSavedVariance_ = device->newBuffer(sizeof(float) * batchSize_ * seqLength_, MTL::ResourceStorageModeManaged);

    // Initialize all to zeros or ones as appropriate:
    memcpy(bufferDebug_->contents(), debug.data(), bufferSize);
    memcpy(bufferGamma_->contents(), gamma.data(), bufferSize);
    memcpy(bufferBeta_->contents(), beta.data(), bufferSize);

    // Also zero out the "saved" stats buffers so they start in a known state
    std::vector<float> zeros(featureDim_, 0.0f);
    memcpy(bufferSavedMean_->contents(), zeros.data(), bufferSize);
    memcpy(bufferSavedVariance_->contents(), zeros.data(), bufferSize);

    // Mark them as modified
    bufferDebug_->didModifyRange(NS::Range(0, bufferSize));
    bufferGamma_->didModifyRange(NS::Range(0, bufferSize));
    bufferBeta_->didModifyRange(NS::Range(0, bufferSize));
    bufferSavedMean_->didModifyRange(NS::Range(0, bufferSize));
    bufferSavedVariance_->didModifyRange(NS::Range(0, bufferSize));
}

void LayerNormalizationLayer::buildBuffers(MTL::Device* device) {
    size_t bufferSize = batchSize_ * seqLength_ * featureDim_ * sizeof(float);

    std::vector<float> gamma(featureDim_, 1.0f); // scale initialized to 1
    std::vector<float> beta(featureDim_, 0.0f);  // shift initialized to 0

    bufferGamma_ = device->newBuffer(sizeof(float) * featureDim_, MTL::ResourceStorageModeManaged);
    bufferBeta_ = device->newBuffer(sizeof(float) * featureDim_, MTL::ResourceStorageModeManaged);

    memcpy(bufferGamma_->contents(), gamma.data(), sizeof(float) * featureDim_);
    memcpy(bufferBeta_->contents(), beta.data(), sizeof(float) * featureDim_);

    bufferGamma_->didModifyRange(NS::Range(0, sizeof(float) * featureDim_));
    bufferBeta_->didModifyRange(NS::Range(0, sizeof(float) * featureDim_));

    // Intermediate buffers for per-sample statistics
    bufferSavedMean_ = device->newBuffer(sizeof(float) * batchSize_ * seqLength_, MTL::ResourceStorageModeManaged);
    bufferSavedVariance_ = device->newBuffer(sizeof(float) * batchSize_ * seqLength_, MTL::ResourceStorageModeManaged);

    // Debug buffer (optional)
    bufferDebug_ = device->newBuffer(sizeof(float) * 256, MTL::ResourceStorageModeManaged);

    // Allocate input and output buffers explicitly (single timestep only)
    inputBuffers_[BufferType::Input].push_back(nullptr);
    outputBuffers_[BufferType::Output].push_back(device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged));
    inputBuffers_[BufferType::IncomingErrors].push_back(nullptr);
    outputBuffers_[BufferType::OutgoingErrors].push_back(device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged));

    // Optimizer buffers for gamma and beta
    optimizerGamma_->buildBuffers(device, featureDim_ * sizeof(float));
    optimizerBeta_->buildBuffers(device, featureDim_ * sizeof(float));
}

void LayerNormalizationLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;

    auto forwardFn = library->newFunction(NS::String::string("forward_layer_norm", NS::UTF8StringEncoding));
    assert(forwardFn && "Forward function not found.");

    auto backwardFn = library->newFunction(NS::String::string("backward_layer_norm", NS::UTF8StringEncoding));
    assert(backwardFn && "Backward function not found.");

    forwardPipelineState_ = device->newComputePipelineState(forwardFn, &error);
    assert(forwardPipelineState_);

    backwardPipelineState_ = device->newComputePipelineState(backwardFn, &error);
    assert(backwardPipelineState_);

    forwardFn->release();
    backwardFn->release();

    ModelConfig* pConfig = ConfigurationManager::instance().getConfig();
    auto optimizerConfig = pConfig->training.optimizer;
    
    uint accumulation_interval = optimizerConfig.accumulation_interval;
    float beta1 = optimizerConfig.beta1;
    float beta2 = optimizerConfig.beta2;
    float epsilon = optimizerConfig.epsilon;

    optimizerGamma_ = std::make_unique<AdamOptimizer>(learningRate_, beta1, beta2, epsilon, accumulation_interval);
    optimizerBeta_  = std::make_unique<AdamOptimizer>(learningRate_, beta1, beta2, epsilon, accumulation_interval);

    optimizerGamma_->buildPipeline(device, library);
    optimizerBeta_->buildPipeline(device, library);
}

void LayerNormalizationLayer::forward(MTL::CommandBuffer* cmdBuf, int batchSize)
{
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);

    encoder->setBuffer(inputBuffers_[BufferType::Input][0], 0, 0);        // input
    encoder->setBuffer(outputBuffers_[BufferType::Output][0], 0, 1);      // output
    encoder->setBuffer(bufferGamma_, 0, 2);
    encoder->setBuffer(bufferBeta_, 0, 3);
    encoder->setBuffer(bufferSavedMean_, 0, 4);
    encoder->setBuffer(bufferSavedVariance_, 0, 5);
    encoder->setBytes(&epsilon_, sizeof(float), 6);
    encoder->setBytes(&featureDim_, sizeof(int), 7);
    encoder->setBytes(&batchSize_, sizeof(int), 8);
    encoder->setBytes(&seqLength_, sizeof(int), 9);
    encoder->setBuffer(bufferDebug_, 0, 10);

    const uint32_t totalThreads = batchSize_ * seqLength_;
    const uint32_t threadsPerGroup = 64;
    uint32_t numThreadgroups = (totalThreads + threadsPerGroup - 1) / threadsPerGroup;

    MTL::Size threadsPerThreadgroup = MTL::Size::Make(threadsPerGroup, 1, 1);
    MTL::Size threadgroups = MTL::Size::Make(numThreadgroups, 1, 1);

    encoder->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
    encoder->endEncoding();
}

void LayerNormalizationLayer::backward(MTL::CommandBuffer* cmdBuf, int batchSize)
{
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);
    
    // indices:
    encoder->setBuffer(inputBuffers_[BufferType::Input][0],       0, 0);
    encoder->setBuffer(inputBuffers_[BufferType::IncomingErrors][0], 0, 1);
    encoder->setBuffer(outputBuffers_[BufferType::OutgoingErrors][0], 0, 2);
    encoder->setBuffer(bufferGamma_, 0, 3);
    encoder->setBuffer(bufferBeta_, 0, 4);
    encoder->setBuffer(bufferSavedMean_, 0, 5);
    encoder->setBuffer(bufferSavedVariance_, 0, 6);
    encoder->setBytes(&epsilon_,       sizeof(float), 7);
    encoder->setBytes(&featureDim_,    sizeof(int),   8);
    encoder->setBytes(&batchSize,      sizeof(uint),  9);
    encoder->setBytes(&seqLength_,     sizeof(uint),  10);
    encoder->setBytes(&learningRate_,  sizeof(float), 11);
    encoder->setBuffer(optimizerBeta_->gradientBuffer(), 0, 12);
    encoder->setBuffer(optimizerGamma_->gradientBuffer(), 0, 13);

    const uint32_t totalThreads = batchSize_ * seqLength_;
    const uint32_t threadsPerGroup = 64;
    uint32_t numThreadgroups = (totalThreads + threadsPerGroup - 1) / threadsPerGroup;

    MTL::Size threadsPerThreadgroup = MTL::Size::Make(threadsPerGroup, 1, 1);
    MTL::Size threadgroups = MTL::Size::Make(numThreadgroups, 1, 1);
    encoder->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
    
    optimizerGamma_->encode(encoder, bufferGamma_, featureDim_, batchSize);
    optimizerBeta_->encode(encoder, bufferBeta_, featureDim_, batchSize);
    
    encoder->endEncoding();
}

void LayerNormalizationLayer::resetErrors() {
    float* errorsBuffer = static_cast<float*>(inputBuffers_[BufferType::IncomingErrors][0]->contents());
    memset(errorsBuffer, 0, inputBuffers_[BufferType::IncomingErrors][0]->length());
    inputBuffers_[BufferType::IncomingErrors][0]->didModifyRange(
        NS::Range::Make(0, inputBuffers_[BufferType::IncomingErrors][0]->length())
    );
}


void LayerNormalizationLayer::updateTargetBufferAt(const float* targetData) {
    assert(false);
}

void LayerNormalizationLayer::updateTargetBufferAt(const float* targetData, int batchSize) {
    assert(false);
}


void LayerNormalizationLayer::setInputBuffer(BufferType type, MTL::Buffer* buffer) {
    assert(buffer && "Setting input buffer to NULL");
    inputBuffers_[type][0] = buffer;
}

MTL::Buffer* LayerNormalizationLayer::getOutputBuffer(BufferType type) {
    return outputBuffers_[type][0];
}

void LayerNormalizationLayer::setOutputBuffer(BufferType type, MTL::Buffer* buffer) {
    outputBuffers_[type][0] = buffer;
}

MTL::Buffer* LayerNormalizationLayer::getInputBuffer(BufferType type) {
    return inputBuffers_[type][0];
}

void LayerNormalizationLayer::connectForwardConnections(Layer* previousLayer) {
    setInputBuffer(BufferType::Input, previousLayer->getOutputBuffer(BufferType::Output));
}

void LayerNormalizationLayer::connectBackwardConnections(Layer* prevLayer) {
    prevLayer->setInputBuffer(BufferType::IncomingErrors, getOutputBuffer(BufferType::OutgoingErrors));
}

void LayerNormalizationLayer::saveParameters(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(bufferGamma_->contents()), bufferGamma_->length());
    os.write(reinterpret_cast<const char*>(bufferBeta_->contents()), bufferBeta_->length());
}

void LayerNormalizationLayer::loadParameters(std::istream& is) {
    is.read(reinterpret_cast<char*>(bufferGamma_->contents()), bufferGamma_->length());
    bufferGamma_->didModifyRange(NS::Range(0, bufferGamma_->length()));

    is.read(reinterpret_cast<char*>(bufferBeta_->contents()), bufferBeta_->length());
    bufferBeta_->didModifyRange(NS::Range(0, bufferBeta_->length()));
}

void LayerNormalizationLayer::onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
    Logger::instance().assertBufferContentsAreValid(outputBuffers_[BufferType::Output][0], getName());
}

void LayerNormalizationLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
    Logger::instance().assertBufferContentsAreValid(outputBuffers_[BufferType::Output][0], getName());
}
