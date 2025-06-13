#include <cstring>
#include <cassert>
#include <algorithm>
#include <iostream>

#include "weight-initializer.h"
#include "configuration-manager.h"
#include "math-lib.h"
#include "adam-optimizer.h"
#include "dense-layer.h"
#include "common.h"
#include "logger.h"

DenseLayer::DenseLayer(int inputDim, int outputDim, int _unused, ActivationFunction activation, int batchSize) :
inputDim_(inputDim),
outputDim_(outputDim),
sequenceLength_(1),
activation_(activation),
bufferWeights_(nullptr),
bufferBias_(nullptr),
bufferGradientScratch_(nullptr),
bufferDeltaScratch_(nullptr),
isTerminal_(false),
initializer_("xavier"),
computeDeltasPipelineState_(nullptr),
forwardPipelineState_(nullptr),
backwardPipelineState_(nullptr),
learningRate_(0.001),
batchSize_(batchSize)
{
    inputBuffers_[BufferType::Input].resize(sequenceLength_, nullptr);
    inputBuffers_[BufferType::IncomingErrors].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::Output].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::OutgoingErrors].resize(sequenceLength_, nullptr);
    inputBuffers_[BufferType::Targets].resize(sequenceLength_, nullptr);
    
}

DenseLayer::~DenseLayer() {
    // Ensure uniqueness before release explicitly
    for (auto ob : outputBuffers_) {
        if (ob.second[0]) {
            ob.second[0]->release();
            ob.second[0] = nullptr;
        }
    }

    if (bufferDeltaScratch_) {
        bufferDeltaScratch_->release();
        bufferDeltaScratch_ = nullptr;
    }
    if (bufferGradientScratch_) {
        bufferGradientScratch_->release();
        bufferGradientScratch_ = nullptr;
    }
    
    if (bufferWeights_) {
        bufferWeights_->release();
        bufferWeights_ = nullptr;
    }
    if (bufferBias_) {
        bufferBias_->release();
        bufferBias_ = nullptr;
    }
    if (forwardPipelineState_) {
        forwardPipelineState_->release();
        forwardPipelineState_ = nullptr;
    }
    if (backwardPipelineState_) {
        backwardPipelineState_->release();
        backwardPipelineState_ = nullptr;
    }
}

void DenseLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    auto forwardFunc = library->newFunction(NS::String::string((activation_ == ActivationFunction::Softmax) ? "forward_softmax_dense_layer" : "forward_non_softmax_dense_layer", NS::UTF8StringEncoding));
    assert(forwardFunc && "Forward function not found.");
    
    auto backwardFunc = library->newFunction(NS::String::string(
                                        (activation_ == ActivationFunction::Softmax) ? (isTerminal_ ? "backward_terminal_softmax_dense_layer" : "backward_non_terminal_softmax_dense_layer") :
                                                                                     (isTerminal_ ? "backward_terminal_non_softmax_dense_layer" : "backward_non_terminal_non_softmax_dense_layer"), NS::UTF8StringEncoding)
                                     );
    assert(backwardFunc && "Backward function not found.");
    
    NS::Error* error = nullptr;
    
    forwardPipelineState_ = device->newComputePipelineState(forwardFunc, &error);
    assert(forwardPipelineState_);
    
    backwardPipelineState_ = device->newComputePipelineState(backwardFunc, &error);
    assert(backwardPipelineState_);
    
    forwardFunc->release();
    backwardFunc->release();
    
    ModelConfig* pConfig = ConfigurationManager::instance().getConfig();
    auto optimizerConfig = pConfig->training.optimizer;
    
    uint accumulation_interval = optimizerConfig.accumulation_interval;
    float beta1 = optimizerConfig.beta1;
    float beta2 = optimizerConfig.beta2;
    float epsilon = optimizerConfig.epsilon;
    
    optimizerWeights_ = std::make_unique<AdamOptimizer>(learningRate_, beta1, beta2, epsilon, accumulation_interval);
    optimizerBiases_  = std::make_unique<AdamOptimizer>(learningRate_, beta1, beta2, epsilon, accumulation_interval);
    
    optimizerWeights_->buildPipeline(device, library);
    optimizerBiases_->buildPipeline(device, library);
}


void DenseLayer::buildBuffers(MTL::Device* device) {
    size_t scratchDeltaSize = outputDim_ * batchSize_ * sizeof(float);
    size_t scratchGradientsSize = inputDim_ * outputDim_ * sizeof(float);
    size_t weightSize = inputDim_ * outputDim_ * sizeof(float);
    size_t biasSize = outputDim_ * sizeof(float);
    
    bufferWeights_ = device->newBuffer(weightSize, MTL::ResourceStorageModeManaged);
    float* w = static_cast<float*>(bufferWeights_->contents());
    if (initializer_ == "he") {
        WeightInitializer::initializeHe(w, inputDim_, outputDim_);
    } else {
        WeightInitializer::initializeXavier(w, inputDim_, outputDim_);
    }
    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    
    bufferBias_ = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    float* b = static_cast<float*>(bufferBias_->contents());
    WeightInitializer::initializeBias(b, outputDim_);
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    
    bufferDeltaScratch_ = device->newBuffer(scratchDeltaSize, MTL::ResourceStorageModeManaged);
    float* ds = static_cast<float*>(bufferDeltaScratch_->contents());
    WeightInitializer::initializeZeros(ds, outputDim_ * batchSize_);
    bufferDeltaScratch_->didModifyRange(NS::Range(0, bufferDeltaScratch_->length()));
    
    bufferGradientScratch_ = device->newBuffer(scratchGradientsSize, MTL::ResourceStorageModeManaged);
    float* gs = static_cast<float*>(bufferGradientScratch_->contents());
    WeightInitializer::initializeZeros(gs, outputDim_ * inputDim_);
    bufferGradientScratch_->didModifyRange(NS::Range(0, bufferGradientScratch_->length()));
    
    outputBuffers_[BufferType::Output].resize(sequenceLength_);
    inputBuffers_[BufferType::IncomingErrors].resize(sequenceLength_);
    outputBuffers_[BufferType::Delta].resize(sequenceLength_);
    inputBuffers_[BufferType::Targets].resize(sequenceLength_);
    outputBuffers_[BufferType::OutgoingErrors].resize(sequenceLength_);
    outputBuffers_[BufferType::Debug].resize(sequenceLength_);
    

    outputBuffers_[BufferType::Delta][0] = device->newBuffer(outputDim_ * batchSize_ * sizeof(float), MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::Delta][0]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Delta][0]->length()));
    
    outputBuffers_[BufferType::OutgoingErrors][0] = device->newBuffer(inputDim_ * batchSize_ * sizeof(float), MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::Output][0] = device->newBuffer(outputDim_ * batchSize_ * sizeof(float), MTL::ResourceStorageModeManaged);
    
    outputBuffers_[BufferType::Debug][0] = device->newBuffer(inputDim_ * outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    inputBuffers_[BufferType::Targets][0] = device->newBuffer(outputDim_ * batchSize_ * sizeof(float), MTL::ResourceStorageModeManaged);
    
    memset(outputBuffers_[BufferType::Output][0]->contents(), 0, outputDim_ * batchSize_ * sizeof(float));
    memset(inputBuffers_[BufferType::Targets][0]->contents(), 0, outputDim_ * batchSize_ * sizeof(float));
    memset(outputBuffers_[BufferType::OutgoingErrors][0]->contents(), 0, inputDim_ * batchSize_ * sizeof(float));
    memset(outputBuffers_[BufferType::Debug][0]->contents(), 0, inputDim_ * outputDim_ * sizeof(float));
    
    inputBuffers_[BufferType::Targets][0]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Targets][0]->length()));
    outputBuffers_[BufferType::Output][0]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Output][0]->length()));
    outputBuffers_[BufferType::OutgoingErrors][0]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::OutgoingErrors][0]->length()));
    outputBuffers_[BufferType::Debug][0]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Debug][0]->length()));
    
    optimizerWeights_->buildBuffers(device, weightSize);
    optimizerBiases_->buildBuffers(device, biasSize);
}

void DenseLayer::setInputBuffer(BufferType type, MTL::Buffer* buffer) {
    assert(buffer && "Setting input buffer to NULL");
    inputBuffers_[type][0] = buffer;
}

void DenseLayer::updateTargetBufferAt(const float* targetData) {
    memcpy(inputBuffers_[BufferType::Targets][0]->contents(), targetData, inputBuffers_[BufferType::Targets][0]->length());
    inputBuffers_[BufferType::Targets][0]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Targets][0]->length()));
}

void DenseLayer::updateTargetBufferAt(const float* targetData, int _batchSize) {
    memcpy(inputBuffers_[BufferType::Targets][0]->contents(), targetData, batchSize_ * outputDim_ * sizeof(float));
    inputBuffers_[BufferType::Targets][0]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Targets][0]->length()));
}

void DenseLayer::forward(MTL::CommandBuffer* cmdBuf, int _batchSize) {
    uint activationRaw = static_cast<uint>(activation_);
    uint bs = (uint)batchSize_;
    uint input_dim = (uint)inputDim_;
    uint output_dim = (uint)outputDim_;
    
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    encoder->setBuffer(inputBuffers_[BufferType::Input][0], 0, 0);
    encoder->setBuffer(outputBuffers_[BufferType::Output][0], 0, 1);
    encoder->setBuffer(bufferWeights_, 0, 2);
    encoder->setBuffer(bufferBias_, 0, 3);
    encoder->setBytes(&input_dim, sizeof(uint), 4);
    encoder->setBytes(&output_dim, sizeof(uint), 5);
    encoder->setBytes(&activationRaw, sizeof(uint), 6);
    encoder->setBytes(&bs, sizeof(uint), 7);
    encoder->setBuffer(outputBuffers_[BufferType::Debug][0], 0, 8);
    
    uint gridSize = batchSize_ * outputDim_;
    uint threadsPerThreadgroup = std::min<uint>(1024, gridSize);
    
    MTL::Size threadgroupSize(threadsPerThreadgroup, 1, 1);
    MTL::Size threadgroups((gridSize + threadsPerThreadgroup - 1) / threadsPerThreadgroup, 1, 1);
    encoder->dispatchThreadgroups(threadgroups, threadgroupSize);
    encoder->endEncoding();
    
    inputBuffers_[BufferType::Input][0]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Input][0]->length()));

}

void DenseLayer::backward(MTL::CommandBuffer* cmdBuf, int _batchSize) {
    uint bs = (uint)batchSize_;
    decay_ *= decayRate_;
    
    auto encoder = cmdBuf->computeCommandEncoder();
    
    inputBuffers_[BufferType::Input][0]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Input][0]->length()));
    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    
    if (isTerminal_ && activation_ != ActivationFunction::Softmax) {
        this->backwardTerminalNonSoftmax(encoder, _batchSize);
    } else if (isTerminal_ && activation_ == ActivationFunction::Softmax) {
        this->backwardTerminalSoftmax(encoder, _batchSize);
    } else if (activation_!=ActivationFunction::Softmax) {
        this->backwardNonTerminalNonSoftmax(encoder, _batchSize);
    } else {
        assert(false && "unsupported layer configuration");
    }
    
    optimizerWeights_->encode(encoder, bufferWeights_, inputDim_ * outputDim_, bs);
    optimizerBiases_->encode(encoder, bufferBias_, outputDim_, bs);
    
    encoder->endEncoding();
}


void DenseLayer::backwardTerminalSoftmax(MTL::ComputeCommandEncoder *encoder, int _batchSize) {
    encoder->setComputePipelineState(backwardPipelineState_);
    
    encoder->setBuffer(inputBuffers_[BufferType::Input][0], 0, 0);                 // inputs
    encoder->setBuffer(bufferWeights_, 0, 1);                                      // weights
    encoder->setBuffer(outputBuffers_[BufferType::Output][0], 0, 2);               // predictions
    encoder->setBuffer(inputBuffers_[BufferType::Targets][0], 0, 3);               // targets
    encoder->setBuffer(optimizerWeights_->gradientBuffer(), 0, 4);                 // gradientWeights
    encoder->setBuffer(optimizerBiases_->gradientBuffer(), 0, 5);                  // gradientBiases
    encoder->setBuffer(outputBuffers_[BufferType::OutgoingErrors][0], 0, 6);       // errorsPreviousLayer

    // Explicitly correct usage of uint for dimensions
    encoder->setBytes(&inputDim_, sizeof(uint), 7);
    encoder->setBytes(&outputDim_, sizeof(uint), 8);
    encoder->setBytes(&batchSize_, sizeof(uint), 9);
    
    const uint TILE_W = 16;
    const uint TILE_H = 16;

    uint groupCountX = (inputDim_ + TILE_W - 1) / TILE_W;
    uint groupCountY = (batchSize_ + TILE_H - 1) / TILE_H;

    MTL::Size threadsPerGroup(TILE_W, TILE_H, 1);
    MTL::Size numThreadgroups(groupCountX, groupCountY, 1);

    encoder->dispatchThreadgroups(numThreadgroups, threadsPerGroup);
    
}

void DenseLayer::backwardTerminalNonSoftmax(MTL::ComputeCommandEncoder *encoder, int _batchSize) {
    uint activationRaw = static_cast<uint>(activation_);

    encoder->setComputePipelineState(backwardPipelineState_);
    
    encoder->setBuffer(inputBuffers_[BufferType::Input][0], 0, 0);
    encoder->setBuffer(bufferWeights_, 0, 1);
    encoder->setBuffer(outputBuffers_[BufferType::Output][0], 0, 2);
    encoder->setBuffer(inputBuffers_[BufferType::Targets][0], 0, 3);
    encoder->setBuffer(optimizerWeights_->gradientBuffer(), 0, 4);
    encoder->setBuffer(optimizerBiases_->gradientBuffer(), 0, 5);
    encoder->setBuffer(outputBuffers_[BufferType::OutgoingErrors][0], 0, 6);
    
    encoder->setBytes(&inputDim_, sizeof(uint), 7);
    encoder->setBytes(&outputDim_, sizeof(uint), 8);
    encoder->setBytes(&batchSize_, sizeof(uint), 9);
    encoder->setBytes(&activationRaw, sizeof(uint), 10);
    
    const uint TILE_W = 16;
    const uint TILE_H = 16;

    uint groupCountX = (outputDim_ + TILE_W - 1) / TILE_W;
    uint groupCountY = (batchSize_ + TILE_H - 1) / TILE_H;

    MTL::Size threadsPerGroup(TILE_W, TILE_H, 1);
    MTL::Size numThreadgroups(groupCountX, groupCountY, 1);

    encoder->dispatchThreadgroups(numThreadgroups, threadsPerGroup);
}

void DenseLayer::backwardNonTerminalNonSoftmax(MTL::ComputeCommandEncoder *encoder, int _batchSize) {
    uint activationRaw = static_cast<uint>(activation_);

    encoder->setComputePipelineState(backwardPipelineState_);
    
    // Binding buffers
    encoder->setBuffer(inputBuffers_[BufferType::Input][0], 0, 0);
    encoder->setBuffer(bufferWeights_, 0, 1);
    encoder->setBuffer(bufferBias_, 0, 2);
    encoder->setBuffer(outputBuffers_[BufferType::Output][0], 0, 3);
    encoder->setBuffer(inputBuffers_[BufferType::IncomingErrors][0], 0, 4);
    encoder->setBuffer(bufferDeltaScratch_, 0, 5);
    encoder->setBytes(&inputDim_, sizeof(uint), 6);
    encoder->setBytes(&outputDim_, sizeof(uint), 7);
    encoder->setBytes(&activationRaw, sizeof(uint), 8);
    encoder->setBytes(&batchSize_, sizeof(uint), 9);
    encoder->setBuffer(outputBuffers_[BufferType::OutgoingErrors][0], 0, 10);
    encoder->setBuffer(optimizerWeights_->gradientBuffer(), 0, 11);
    encoder->setBuffer(optimizerBiases_->gradientBuffer(), 0, 12);

    uint gridSize = batchSize_ * outputDim_;
    uint threadsPerThreadgroup = std::min<uint>(1024, gridSize);
    
    MTL::Size threadgroupSize(threadsPerThreadgroup, 1, 1);
    MTL::Size threadgroups((gridSize + threadsPerThreadgroup - 1) / threadsPerThreadgroup, 1, 1);
    encoder->dispatchThreadgroups(threadgroups, threadgroupSize);
}

void DenseLayer::setOutputBuffer(BufferType type, MTL::Buffer* buffer) {
    outputBuffers_[type][0] = buffer;
}

MTL::Buffer* DenseLayer::getInputBuffer(BufferType type) {
    return inputBuffers_[type][0];
}

MTL::Buffer* DenseLayer::getOutputBuffer(BufferType type) {
    return outputBuffers_[type][0];
}

int DenseLayer::inputSize() const {
    return inputDim_;
}

int DenseLayer::outputSize() const {
    return outputDim_;
}

void DenseLayer::resetErrors() {
    float* errorsBuffer = static_cast<float*>(outputBuffers_[BufferType::OutgoingErrors][0]->contents());
    memset(errorsBuffer, 0, outputBuffers_[BufferType::OutgoingErrors][0]->length());
    outputBuffers_[BufferType::OutgoingErrors][0]->didModifyRange(
        NS::Range::Make(0, outputBuffers_[BufferType::OutgoingErrors][0]->length())
    );
}

void DenseLayer::connectForwardConnections(Layer* previousLayer) {
    setInputBuffer(BufferType::Input, previousLayer->getOutputBuffer(BufferType::Output));
}

void DenseLayer::connectBackwardConnections(Layer* prevLayer)
{
    prevLayer->setInputBuffer(BufferType::IncomingErrors, getOutputBuffer(BufferType::OutgoingErrors));
}

void DenseLayer::saveParameters(std::ostream& os) const { //FIXME encode buffer lengths
    os.write(reinterpret_cast<const char*>(bufferWeights_->contents()), bufferWeights_->length());
    os.write(reinterpret_cast<const char*>(bufferBias_->contents()), bufferBias_->length());
}

void DenseLayer::loadParameters(std::istream& is) { //FIXME - decode buffer lengths and verify
    is.read(reinterpret_cast<char*>(bufferWeights_->contents()), bufferWeights_->length());
    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    
    is.read(reinterpret_cast<char*>(bufferBias_->contents()), bufferBias_->length());
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
}

void DenseLayer::onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {}


void DenseLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
    
}

void DenseLayer::debugLog() {
}
