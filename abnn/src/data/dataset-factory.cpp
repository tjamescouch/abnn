//
//  dataset-factory.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-21.
//

#include "dataset-factory.h"
#include "mnist-dataset.h"
#include "function-dataset.h"
#include "text-crawler.h"
#include "tokenized-dataset.h"
#include "character-tokenizer.h"
#include "model-config.h"
#include <stdexcept>
#include "math-lib.h"
#include <memory>

// Explicitly creates dataset based on model configuration
Dataset* DatasetFactory::createDataset(const ModelConfig* pConfig) {
    if (pConfig->dataset.type == "mnist") {
        return new MNISTDataset(
            pConfig->dataset.images,
            pConfig->dataset.labels,
            pConfig->training.batch_size
        );
    } else if (pConfig->dataset.type == "function") {
        int inputShape[2] = {};
        pConfig->layers.front().params.at("output_shape").get_value_inplace(inputShape);
        int inputSequenceLength = inputShape[0];
        int targetSequenceLength = 1;
        int featureDim = inputShape[1];
        int datasetSize = pConfig->dataset.dataset_size;

        int outputDim = 0;
        if (pConfig->layers.back().params.contains("output_shape")) {
            int outputShape[2] = {};
            pConfig->layers.back().params.at("output_shape").get_value_inplace(outputShape);
            targetSequenceLength = outputShape[0];
            outputDim = outputShape[1];
        } else {
            outputDim = pConfig->layers.back().params.at("output_size").get_value<int>();
        }

        return new FunctionDataset(mathlib::inputFunc, mathlib::targetFunc,
                                   inputSequenceLength,
                                   targetSequenceLength,
                                   featureDim,
                                   outputDim,
                                   datasetSize);
    } else if (pConfig->dataset.type == "text") {
        // Extract parameters explicitly from YAML
        const std::string& corpusDirectory = pConfig->dataset.corpus_directory;
        int sequenceLength = pConfig->dataset.sequence_length;
        int samplesPerFile = pConfig->dataset.samples_per_file;
        int batchSize = pConfig->training.batch_size;

        auto crawler = std::make_unique<TextCrawler>(corpusDirectory, sequenceLength, samplesPerFile);

        if (pConfig->dataset.tokenizer.type == "character") {
            auto tokenizer = std::make_unique<CharacterTokenizer>();

            return new TokenizedDataset(crawler.release(), tokenizer.release(), sequenceLength, batchSize);
        } else {
            throw std::runtime_error("Unsupported tokenizer type: " + pConfig->dataset.tokenizer.type);
        }
    } else {
        throw std::runtime_error("Unsupported dataset type: " + pConfig->dataset.type);
    }
}
