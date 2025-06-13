#pragma once

#include <functional>
#include <string>
#include <memory> // explicitly include for unique_ptr

#include "dataset.h"
#include "model-config.h"

class DataManager {
public:
    DataManager();
    ~DataManager();

    void setDataset(std::unique_ptr<Dataset> dataset); // explicitly updated parameter
    Dataset* getCurrentDataset() const;

    DataManager* configure(ModelConfig* pConfig);
    void initialize(int batchSize, std::function<void()> callback);

    int inputDim() const;
    int outputDim() const;
    void loadNextBatch(int currentBatchSize);

private:
    std::unique_ptr<Dataset> dataset_; // explicitly manage dataset lifetime safely
    int sampleIndex_ = 0;
};
