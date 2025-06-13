#ifndef DATASET_H
#define DATASET_H

class Dataset {
public:
    // Required methods:
    virtual void loadData(int batchSize) = 0;
    virtual ~Dataset() {}

    virtual const float* getInputDataAt(int batchIndex) const = 0;
    virtual const float* getTargetDataAt(int batchIndex) const = 0;
    
    virtual int numSamples() const = 0;

    virtual int getDatasetSize() const = 0;
    virtual float calculateLoss(const float* predictedData, int outputDim, const float* targetData, int currentBatchSize, const float* inputData, int inputSize) = 0;


    virtual int inputDim() const = 0;
    virtual int outputDim() const = 0;
    
    virtual void loadNextBatch(int batchSize) = 0;
};

#endif
