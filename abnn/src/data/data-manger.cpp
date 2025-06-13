//
//  data-manger.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//
#include <stdexcept>
#include <memory>

#include "data-manager.h"
#include <cassert>
#include "math-lib.h"
#include "function-dataset.h"
#include "mnist-dataset.h"
#include "dataset-factory.h"


DataManager::DataManager()
: dataset_(nullptr) {
}

DataManager* DataManager::configure(ModelConfig* pConfig) {
    dataset_ = std::unique_ptr<Dataset>(DatasetFactory::createDataset(pConfig)); // explicitly use smart pointer
    return this;
}

DataManager::~DataManager() = default; // explicitly use default destructor (smart pointer handles deletion)

void DataManager::setDataset(std::unique_ptr<Dataset> dataset) {
    dataset_ = std::move(dataset); // explicitly transfer ownership safely
}

Dataset* DataManager::getCurrentDataset() const {
    if (!dataset_) {
        throw std::runtime_error("Dataset has not been set.");
    }
    return dataset_.get(); // explicitly return raw pointer
}

void DataManager::initialize(int batchSize, std::function<void()> callback) {
    if (!dataset_) {
        throw std::runtime_error("Cannot initialize DataManager: no dataset set.");
    }
    
    //dataset_->loadData(batchSize);
    callback();
}

int DataManager::inputDim() const {
    return dataset_->inputDim();
}

int DataManager::outputDim() const {
    return dataset_->outputDim();
}

void DataManager::loadNextBatch(int currentBatchSize) {
    dataset_->loadNextBatch(currentBatchSize);
}
