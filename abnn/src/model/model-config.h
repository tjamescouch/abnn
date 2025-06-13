//
//  model-config.h
//  MetalNeuron
//
//  Created by James Couch on 2025-02-28.
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include <fkYAML/node.hpp>

// Layer configuration
struct LayerConfig {
    std::string type;
    std::map<std::string, fkyaml::node> params;
    int time_steps = -1;
    float learning_rate;
};

// Optimizer configuration
struct OptimizerConfig {
    std::string type;
    uint accumulation_interval;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    std::map<std::string, fkyaml::node> parameters;
};

// Training configuration
struct TrainingConfig {
    OptimizerConfig optimizer;
    int epochs;
    int batch_size;
};

struct ModelTokenizer {
    std::string type;
    struct Parameters {
        int vocab_size;
        int embedding_dim;
    } parameters;
};

struct ModelDataSet {
    std::string type;
    std::string images;
    std::string labels;
    uint dataset_size;
    ModelTokenizer tokenizer; // explicitly added tokenizer struct
    std::string corpus_directory;  // explicitly added for text datasets
    int sequence_length;           // explicitly added for text datasets
    int samples_per_file;          // explicitly added for text datasets
};


// Overall model configuration
class ModelConfig {
public:
    int first_layer_time_steps = 1;
    std::string name;
    std::vector<LayerConfig> layers;
    TrainingConfig training;
    std::map<std::string, fkyaml::node> metadata;
    ModelDataSet dataset;
    std::string filename;
    
    static ModelConfig loadFromFile(const std::string& filePath);
};
