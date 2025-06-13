#include "tokenized-dataset.h"
#include <algorithm>
#include <cassert>
#include <numeric>
#include "logger.h"

// Constructor explicitly initializes members and loads initial batch
TokenizedDataset::TokenizedDataset(TextCrawler* textCrawler, Tokenizer* tokenizer,
                                   int sequenceLength, int batchSize)
: textCrawler_(textCrawler), tokenizer_(tokenizer),
sequenceLength_(sequenceLength), batchSize_(batchSize),
currentBatchIndex_(0) {
}

TokenizedDataset::~TokenizedDataset() {}

// explicitly number of total samples (here infinite, but return large fixed value)
int TokenizedDataset::numSamples() const {
    return 1000; // large explicit number for continuous training
}

int TokenizedDataset::getDatasetSize() const {
    return batchSize_;
}

// explicitly get flattened input data for batch
const float* TokenizedDataset::getInputDataAt(int batchIndex) const {
    assert(batchIndex < batchSize_);
    return &flattenedInputBuffer_[batchIndex * sequenceLength_];
}

// explicitly get flattened target data for batch
const float* TokenizedDataset::getTargetDataAt(int batchIndex) const {
    assert(batchIndex < batchSize_);
    return &flattenedTargetBuffer_[batchIndex * tokenizer_->vocabSize()];
}

// explicitly tokenizes new batch of raw text
void TokenizedDataset::loadData(int _batchSize) {
    inputData_.resize(batchSize_);
    targetData_.resize(batchSize_);
    
    for (int i = 0; i < batchSize_; ++i) {
        std::string sequence = textCrawler_->getRandomSequence();
        std::vector<int> tokenIds = tokenizer_->tokenize(sequence);
        
        assert(tokenIds.size() == sequenceLength_ + 1);
        
        inputData_[i].resize(0);
        targetData_[i].resize(0);
        
        for (int iToken = 0; iToken < tokenIds.size(); iToken++){
            int token = tokenIds[iToken];
            
            if (iToken == tokenIds.size() - 1) {
                targetData_[i].push_back(token);   // targets explicitly next token predictions
            } else {
                inputData_[i].push_back(token);    // inputs explicitly all tokens except last
            }
        }
    }
    
    preprocessBatch();
}

// explicitly shuffles batch indices (for completeness, though batches are random already)
void TokenizedDataset::shuffleIndices() {
    shuffledIndices_.resize(batchSize_);
    std::iota(shuffledIndices_.begin(), shuffledIndices_.end(), 0);
    std::shuffle(shuffledIndices_.begin(), shuffledIndices_.end(), std::mt19937(std::random_device{}()));
}

// explicitly prepares flattened buffers for neural network consumption
void TokenizedDataset::preprocessBatch() {
    int vocabSize = (int)tokenizer_->vocabSize();
    
    flattenedInputBuffer_.resize(0);
    flattenedTargetBuffer_.resize(0);
    
    flattenedInputBuffer_.resize(batchSize_ * (sequenceLength_));
    flattenedTargetBuffer_.resize(batchSize_ * (sequenceLength_) * vocabSize);
    std::fill(flattenedTargetBuffer_.begin(), flattenedTargetBuffer_.end(), 0.0f);

    for (int i = 0; i < batchSize_; ++i) {
        // copy inputs explicitly
        std::copy(inputData_[i].begin(), inputData_[i].end(), flattenedInputBuffer_.begin() + i * (sequenceLength_));

        // explicitly one-hot encode targets
        int tokenID = targetData_[i][0];
        oneHotEncode(flattenedTargetBuffer_, (i), vocabSize, tokenID);
    }
}

void TokenizedDataset::loadNextBatch(int currentBatchSize) {
    assert(currentBatchSize <= batchSize_);
    loadData(currentBatchSize);
}

float TokenizedDataset::calculateLoss(const float* predictions, int outputDim, const float* targets, int currentBatchSize, const float* inputData, int inputSize) {
    float loss = 0.0f;
    int dim = this->outputDim();
    
    for (int batch = 0; batch < currentBatchSize; ++batch) {
        std::vector<int> v;
        for (int i = 0; i < sequenceLength_; i++) {
            v.push_back(inputData[i + batch * sequenceLength_]);
        }
        std::string s = tokenizer_->detokenize(v);
        
        int targetTokenId = probabilityDecode(targets, batch, dim);
        int predictedTokenId = probabilityDecode(predictions, batch, dim);


        std::string predictedToken = tokenizer_->detokenize({predictedTokenId});
        std::string targetToken = tokenizer_->detokenize({targetTokenId});

        predictedToken = predictedToken == "\n" ? "\\n" : predictedToken;
        targetToken = targetToken == "\n" ? "\\n" : targetToken;

        if (predictedToken == targetToken) {
            Logger::log << "💎 '" << predictedToken << "'" << std::endl;
            Logger::log << "'" << s << predictedToken << "'" << std::endl;
        } else {
            Logger::log << "❌ predicted: '" << predictedToken << "'" << std::endl;
            Logger::log << "🟢 target:    '" << targetToken << "'" << std::endl;
            Logger::log << "predicted: '" << s << predictedToken << "'" << std::endl;
            Logger::log << "target:    '" << s << targetToken << "'" << std::endl;
        }

        // Correct cross-entropy loss calculation using one-hot encoded targets explicitly
        int correctClassIndex = batch * dim + targetTokenId;
        float pred = predictions[correctClassIndex];
        loss += -logf(pred + 1e-9f);

    }
    
    return loss / static_cast<float>(currentBatchSize);
}

int TokenizedDataset::inputDim() const {
    // input dimension equals sequence length minus 1 (since the last token is used as target)
    return sequenceLength_;
}

int TokenizedDataset::outputDim() const {
    // output dimension is the vocabulary size (number of possible token IDs)
    return (int)tokenizer_->vocabSize();
}

void TokenizedDataset::oneHotEncode(std::vector<float>& buffer, int index, int vocabSize, int tokenID) {
    int offset = index * vocabSize;
    buffer[offset + tokenID] = 1.0f;
}

int TokenizedDataset::probabilityDecode(const float* vector, int index, int vocabSize) {
    int offset = index * vocabSize;
    int maxTokenID = 0;
    float maxValue = 0.0f;

    for (int tokenID = 0; tokenID < vocabSize; ++tokenID) {
        float value = (vector[offset + tokenID]);
        if (value > maxValue) {
            maxTokenID = tokenID;
            maxValue = value;
        }
    }

    return maxTokenID;
}
