//
//  model-config.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-02-28.
//
#include "model-config.h"
#include <fkYAML/node.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

ModelConfig ModelConfig::loadFromFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file) {
        throw std::runtime_error("Could not open YAML file: " + filePath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    auto config = fkyaml::node::deserialize(buffer.str());
    ModelConfig modelConfig;
    
    
    // Load optimizer
    const auto optimizerNode = config["training"]["optimizer"];
    OptimizerConfig optimizerConfig;
    optimizerConfig.type = optimizerNode["type"].get_value<std::string>();
    
    float deault_learning_rate = 1e-4;
    float default_beta1 = 0.9f;
    float default_beta2 = 0.999f;
    float epsilon = 1e-8;
    
    optimizerConfig.accumulation_interval = optimizerNode["accumulation_interval"].get_value_or<uint>(1);
    optimizerConfig.learning_rate = optimizerNode["learning_rate"].get_value_or<float>(deault_learning_rate);
    optimizerConfig.beta1   = optimizerNode["parameters"]["beta1"].get_value_or<float>(default_beta1);
    optimizerConfig.beta2   = optimizerNode["parameters"]["beta2"].get_value_or<float>(default_beta2);
    optimizerConfig.epsilon = optimizerNode["parameters"]["epsilon"].get_value_or<float>(epsilon);
    
    optimizerConfig.accumulation_interval = optimizerConfig.accumulation_interval > 0 ? optimizerConfig.accumulation_interval : 1;
    optimizerConfig.learning_rate = optimizerConfig.learning_rate > 0 ? optimizerConfig.learning_rate : deault_learning_rate;
    optimizerConfig.beta1 = optimizerConfig.beta1 > 0 ? optimizerConfig.beta1 : default_beta1;
    optimizerConfig.beta2 = optimizerConfig.beta2 > 0 ? optimizerConfig.beta2 : default_beta2;
    optimizerConfig.epsilon = optimizerConfig.epsilon > 0 ? optimizerConfig.epsilon : epsilon;

    if (optimizerNode.contains("parameters")) {
        for (const auto& param : optimizerNode["parameters"].as_map()) {
            optimizerConfig.parameters[param.first.get_value<std::string>()] = param.second;
        }
    }
    
    // Load training details
    modelConfig.training.optimizer = optimizerConfig;
    modelConfig.training.epochs = config["training"]["epochs"].get_value<int>();
    modelConfig.training.batch_size = config["training"]["batch_size"].get_value<int>();
    
    

    // Load basic fields
    modelConfig.name = config["name"].get_value<std::string>();
    modelConfig.dataset.type = config["dataset"]["type"].get_value<std::string>();
    int defaultDatasetSize = 1000;
    if (config["dataset"].contains("dataset_size")) {
        modelConfig.dataset.dataset_size = config["dataset"]["dataset_size"].get_value_or<int>(defaultDatasetSize);
    }
    modelConfig.dataset.dataset_size = modelConfig.dataset.dataset_size > 0 ? modelConfig.dataset.dataset_size : defaultDatasetSize;
    
    if (modelConfig.dataset.type == "mnist") {
        modelConfig.dataset.labels = config["dataset"]["labels"].get_value<std::string>();
        modelConfig.dataset.images = config["dataset"]["images"].get_value<std::string>();
    }

    if (modelConfig.dataset.type == "text") {
        modelConfig.dataset.corpus_directory = config["dataset"]["corpus_directory"].get_value<std::string>();
        modelConfig.dataset.sequence_length = config["dataset"]["sequence_length"].get_value<int>();
        modelConfig.dataset.samples_per_file = config["dataset"]["samples_per_file"].get_value<int>();
        const auto tokenizerNode = config["dataset"]["tokenizer"];
        modelConfig.dataset.tokenizer.type = tokenizerNode["type"].get_value<std::string>();
        modelConfig.dataset.tokenizer.parameters.vocab_size = tokenizerNode["parameters"]["vocab_size"].get_value<int>();
        modelConfig.dataset.tokenizer.parameters.embedding_dim = tokenizerNode["parameters"]["embedding_dim"].get_value<int>();
    }
    
    bool isFirstLayer = true;
    // Load layers
    for (const auto& layer : config["layers"]) {
        LayerConfig layerConfig;
        layerConfig.type = layer["type"].get_value<std::string>();
        layerConfig.learning_rate = layer["learning_rate"].get_value_or<float>(optimizerConfig.learning_rate);
        layerConfig.learning_rate = layerConfig.learning_rate > 0 ? layerConfig.learning_rate : optimizerConfig.learning_rate;
        
        layerConfig.time_steps = layer["time_steps"].get_value<int>();
        
        if (isFirstLayer) {
            isFirstLayer = false;
            int first_layer_time_steps = layer["time_steps"].get_value<int>();
            if (first_layer_time_steps > 0){
                modelConfig.first_layer_time_steps = first_layer_time_steps;
            }
        }

        // Load layer parameters
        for (const auto& param : layer.as_map()) {
            const std::string& key = param.first.get_value<std::string>();
            if (key != "type") {
                layerConfig.params[key] = param.second;
            }
        }

        modelConfig.layers.push_back(layerConfig);
    }

    // Load metadata (optional)
    if (config.contains("metadata")) {
        for (const auto& item : config["metadata"].as_map()) {
            modelConfig.metadata[item.first.get_value<std::string>()] = item.second;
        }
    }

    return modelConfig;
}
