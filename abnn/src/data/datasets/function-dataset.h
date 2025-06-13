#pragma once

#include "dataset.h"
#include <functional>
#include <vector>
#include <numeric> 
#include <algorithm>
#include <random>

using InputFunction = std::function<float(int, double)>;
using TargetFunction = std::function<float(int, double)>;

class FunctionDataset : public Dataset {
public:
    FunctionDataset(InputFunction inputFunc, TargetFunction targetFunc,
                    int inputSequenceLength, int targetSequenceLength,
                    int inputDim, int outputDim, int datasetSize);
    ~FunctionDataset() override;

    void loadData(int batchSize) override;
    
    const float* getInputDataAt(int batchIndex) const override;
    const float* getTargetDataAt(int batchIndex) const override;
    
    float calculateLoss(const float* predictedData, int outputDim, const float* targetData, int currentBatchSize, const float* inputData, int inputSize) override;

    int getDatasetSize() const override;
    
    int inputDim() const override { return inputDim_; };
    int outputDim() const override { return outputDim_; };
    
    void loadNextBatch(int batchSize) override;

    int numSamples() const override;
    
private:
    InputFunction inputFunc_;
    TargetFunction targetFunc_;
    int inputSequenceLength_;
    int targetSequenceLength_;
    int inputDim_;
    int outputDim_;
    int datasetSize_;
    int offset_ = 0;

    std::vector<int> shuffledIndices_;
    std::vector<float> inputs_;
    std::vector<float> targets_;
    
    void shuffleIndices();
    void generateBatch(double offset, int batchSize);
};
