#pragma once

#include "dataset.h"
#include <vector>
#include <string>

class MNISTDataset : public Dataset {
public:
    MNISTDataset(const std::string& imagesFilename, const std::string& labelsFilename, int batchSize);
    ~MNISTDataset() override;

    // Overrides from Dataset interface
    void loadData(int batchSize) override;
    
    const float* getInputDataAt(int batchIndex) const override;
    const float* getTargetDataAt(int batchIndex) const override;
    
    float calculateLoss(const float* predictedData, int outputDim, const float* targetData, int currentBatchSize, const float* inputData, int inputSie) override;

    int getDatasetSize() const override;

    // Existing specific methods
    int numSamples() const override;
    int inputDim() const override;
    int outputDim() const override;

    const std::vector<float>& inputAt(int index);
    const std::vector<float>& targetAt(int index);
    
    void loadNextBatch(int batchSize) override;

private:
    void loadImages(const std::string& imagesPath);
    void loadLabels(const std::string& labelsPath);

    std::vector<std::vector<float>> inputs_;
    std::vector<std::vector<float>> targets_;
    
    int batchSize_ = 1;
    int pageOffset_ = 0;
    
    float* batchedInputData_;
    float* batchedTargetData_;

    int currentSampleIndex_;

    std::vector<float> currentInputBuffer_;
    std::vector<float> currentTargetBuffer_;
};
