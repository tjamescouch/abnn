#pragma once

#include "dataset.h"
#include "text-crawler.h"
#include "tokenizer.h"
#include <vector>

class TokenizedDataset : public Dataset {
public:
    TokenizedDataset(TextCrawler* textCrawler, Tokenizer* tokenizer,
                     int sequenceLength, int batchSize);
    virtual ~TokenizedDataset();

    int getDatasetSize() const override;
    const float* getInputDataAt(int batchIndex) const override;
    const float* getTargetDataAt(int batchIndex) const override;

    void loadData(int batchSize) override;
    int numSamples() const override;
    int inputDim() const override;
    int outputDim() const override;
    void loadNextBatch(int currentBatchSize) override;
    float calculateLoss(const float* predictedData, int outputDim, const float* targetData, int currentBatchSize, const float* inputData, int inputSie) override;


private:
    void oneHotEncode(std::vector<float>& buffer, int index, int vocabSize, int tokenID);
    int probabilityDecode(const float* vector, int index, int vocabSize);
    void shuffleIndices();
    void preprocessBatch(); // explicitly tokenize raw sequences into numeric batches

    TextCrawler* textCrawler_;        // explicitly raw text provider
    Tokenizer* tokenizer_;            // explicitly tokenizer instance

    std::vector<std::vector<float>> inputData_;   // explicitly numeric inputs (token IDs as floats)
    std::vector<std::vector<float>> targetData_;  // explicitly numeric targets (next-token predictions)

    int sequenceLength_;
    int batchSize_;
    std::vector<int> shuffledIndices_;

    std::vector<float> flattenedInputBuffer_;  // explicitly flattened buffers for neural net
    std::vector<float> flattenedTargetBuffer_;

    int currentBatchIndex_; // explicitly tracks current batch position
};
