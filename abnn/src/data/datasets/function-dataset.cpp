#include <random>
#include <stdexcept>

#include "math-lib.h"
#include "training-manager.h"
#include "function-dataset.h"



std::default_random_engine generator;
std::uniform_int_distribution<int> distribution(0, 200*M_PI);


FunctionDataset::FunctionDataset(InputFunction inputFunc, TargetFunction targetFunc,
                                 int inputSequenceLength, int targetSequenceLength, int inputDim, int outputDim, int datasetSize)
: inputFunc_(inputFunc),
  targetFunc_(targetFunc),
  inputSequenceLength_(inputSequenceLength),
  targetSequenceLength_(targetSequenceLength),
  inputDim_(inputDim),
  outputDim_(outputDim),
  datasetSize_(datasetSize),
  inputs_(0),
  targets_(0) {
}

FunctionDataset::~FunctionDataset() {
    
}

int FunctionDataset::getDatasetSize() const {
    return datasetSize_;
}

float FunctionDataset::calculateLoss(const float* predictedData, int outputDim, const float* targetData, int currentBatchSize, const float* inputData, int inputSize) {
    float mse = 0.0f;

    for (int i = 0; i < outputDim; ++i) {
        float diff = predictedData[i] - targetData[i];
        mse += diff * diff;
    }

    mse /= static_cast<float>(outputDim);
    return mse;
}



void FunctionDataset::loadData(int batchSize) {
    bool isTraining = TrainingManager::instance().isTraining();
    if (isTraining) {
        shuffleIndices();
        generateBatch(offset_, batchSize);
    } else {
        generateBatch(offset_, batchSize);
        offset_ += inputSequenceLength_; // explicitly increment offset for continuous movement
    }
}

void FunctionDataset::generateBatch(double baseOffset, int batchSize) {
    const int inputBatchDataSize = batchSize * inputSequenceLength_ * inputDim_;
    inputs_.resize(inputBatchDataSize);

    const int targetBatchDataSize = batchSize * targetSequenceLength_ * outputDim_;
    targets_.resize(targetBatchDataSize);

    bool isTraining = TrainingManager::instance().isTraining();

    for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        // Explicitly set stable offset during evaluation to prevent flutter
        double sampleOffset = isTraining ? baseOffset + batchIdx * inputSequenceLength_ : baseOffset;

        // Generate inputs explicitly per timestep
        for (int seqIdx = 0; seqIdx < inputSequenceLength_; ++seqIdx) {
            for (int dim = 0; dim < inputDim_; ++dim) {
                int inputIndex = batchIdx * (inputSequenceLength_ * inputDim_) 
                                 + seqIdx * inputDim_ + dim;
                inputs_[inputIndex] = inputFunc_(dim, sampleOffset + seqIdx);
            }
        }

        // Generate targets explicitly aligned to input sequences
        for (int seqIdx = 0; seqIdx < targetSequenceLength_; ++seqIdx) {
            for (int dim = 0; dim < outputDim_; ++dim) {
                int targetIndex = batchIdx * (targetSequenceLength_ * outputDim_)
                                  + seqIdx * outputDim_ + dim;
                targets_[targetIndex] = targetFunc_(dim, sampleOffset + inputSequenceLength_ + seqIdx);
            }
        }
    }
}

void FunctionDataset::loadNextBatch(int batchSize) {
    loadData(batchSize);
}

void FunctionDataset::shuffleIndices() {
    offset_ = (int)round(distribution(generator));
}

int FunctionDataset::numSamples() const {
    // Returns the total number of samples in this dataset
    return datasetSize_;
}


const float* FunctionDataset::getInputDataAt(int batchIndex) const {
    return inputs_.data() + batchIndex * inputDim_ * inputSequenceLength_;
}

const float* FunctionDataset::getTargetDataAt(int batchIndex) const {
    return targets_.data() + batchIndex * outputDim_;
}
