//
//  layer-factory.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-04.
//
#include <iostream>
#include "layer-factory.h"
#include "dense-layer.h"
#include "dropout-layer.h"
#include "multi-head-attention-layer.h"
#include "batch-normalization-layer.h"
#include "layer-normalization-layer.h"
#include "residual-connection-layer.h"
#include "embedding-layer.h"
#include "positional-encoding-layer.h"
#include "self-attention-layer.h"
#include "flatten-layer.h"
#include "reshape-layer.h"
#include "map-reduce-layer.h"
#include "configuration-manager.h"
#include "logger.h"

const char* inputLayerName = "input";

LayerFactory::LayerFactory() {
}

Layer* LayerFactory::createLayer(LayerConfig& layerConfig,
                                 MTL::Device* device,
                                 MTL::Library* library,
                                 bool isTerminal) {
    Logger::log << "Getting layer name" << std::endl;
    // Provide a default sequential numeric ID if name not explicitly provided
    std::string layerName = layerConfig.params["name"].get_value_or<std::string>(
        "layer_" + std::to_string(layerIdCounter_++)
    );
    layerConfig.params["name"] = layerName;
    
    auto initializer = layerConfig.params["initializer"].get_value_or<std::string>("xavier");
    
    Logger::log << "Configuring layer" << layerName << std::endl;

    Logger::log << "Getting global parameters..." << std::endl;
    auto config = ConfigurationManager::instance().getConfig();
    auto batchSize = config->training.batch_size;

    Logger::log << "Getting common layer parameters..." << std::endl;
    int inputSize = 0;
    int outputSize = 0;
    int sequenceLength = 1;  // default for non-sequence layers
    int outputSequenceLength = 1;  // default for non-sequence layers

    Logger::log << "Getting input shape" << std::endl;
    // Explicitly handle shapes for sequence-aware layers
    if (layerConfig.params.contains("input_shape")) {
        int inputShape[2] = {};
        layerConfig.params["input_shape"].get_value_inplace(inputShape);
        sequenceLength = inputShape[0];
        inputSize = inputShape[1];
    } else {
        inputSize = layerConfig.params["input_size"].get_value<int>();
    }

    Logger::log << "Getting output shape" << std::endl;
    if (layerConfig.params.contains("output_shape")) {
        int outputShape[2] = {};
        layerConfig.params["output_shape"].get_value_inplace(outputShape);
        outputSequenceLength = outputShape[0];
        outputSize = outputShape[1];
    } else {
        outputSize = layerConfig.params["output_size"].get_value<int>();
    }

    auto learningRate = layerConfig.learning_rate;

    Layer* layer = nullptr;

    if (layerConfig.type == "Embedding") {
        Logger::log << "Creating embedding layer..." << std::endl;

        int vocabSize = layerConfig.params.at("vocab_size").get_value<int>();
        int embeddingDim = layerConfig.params.at("embedding_dim").get_value<int>();
        int sequenceLength = layerConfig.params.at("input_size").get_value<int>();
        assert(sequenceLength == outputSequenceLength);

        layer = (new EmbeddingLayer(vocabSize, embeddingDim, sequenceLength, outputSize, batchSize))->setInitializer(initializer)->setLearningRate(learningRate);

    } else if (layerConfig.type == "Input") {
        Logger::log << "Creating input layer..." << std::endl;

        // Instantiate the InputLayer explicitly with sequence awareness
        layer = new InputLayer(outputSequenceLength, outputSize, batchSize);
    } else if (layerConfig.type == "Dense") {
        Logger::log << "Creating dense layer..." << std::endl;
        auto activationStr = layerConfig.params.at("activation").get_value<std::string>();

        ActivationFunction activation = parseActivation(activationStr);
        layer = (new DenseLayer(inputSize, outputSize, 1, activation, batchSize))
                    ->setLearningRate(learningRate)
                    ->setInitializer(initializer);
        
    } else if (layerConfig.type == "Dropout") {
        Logger::log << "Creating dropout layer..." << std::endl;
        float rate = layerConfig.params.at("rate").get_value_or<float>(0.3);
        layer = new DropoutLayer(rate, inputSize, outputSize, batchSize, 1);
        
    } else if (layerConfig.type == "SelfAttention") {
        Logger::log << "Creating self attention layer..." << std::endl;
        
        auto initializer = layerConfig.params["initializer"].get_value_or<std::string>("xavier");
        
        layer = (new SelfAttentionLayer(inputSize, outputSize, sequenceLength, batchSize))->setInitializer(initializer);
        
    } else if (layerConfig.type == "MultiHeadAttention") {
        Logger::log << "Creating multi-head attention layer..." << std::endl;
        int num_heads = layerConfig.params.at("num_heads").get_value_or<int>(2);
        auto initializer = layerConfig.params["initializer"].get_value_or<std::string>("xavier");
        
        layer = (new MultiHeadAttentionLayer(inputSize, outputSize, sequenceLength, batchSize, num_heads))->setInitializer(initializer);
        
    } else if (layerConfig.type == "BatchNormalization") {
        Logger::log << "Creating batch normalization layer..." << std::endl;
        float epsilon = layerConfig.params["epsilon"].get_value_or<float>(1e-5f);
        epsilon = epsilon > 0 ? epsilon : 1e-5f;
        layer = new BatchNormalizationLayer(inputSize, outputSize, batchSize, 1, learningRate, epsilon);
        
    } else if (layerConfig.type == "LayerNormalization") {
        Logger::log << "Creating layer normalization layer..." << std::endl;
        float epsilon = layerConfig.params["epsilon"].get_value_or<float>(1e-5f);
        epsilon = epsilon > 0 ? epsilon : 1e-5f;
        layer = new LayerNormalizationLayer(inputSize, sequenceLength, batchSize, learningRate, epsilon);
        
    } else if (layerConfig.type == "ResidualConnection") {
        auto from = layerConfig.params.at("from_layer").get_value<std::string>();
        float scale = layerConfig.params["scale"].get_value_or<float>(1);
        scale = scale > 0 ? scale : 1;
        
        Logger::log << "Creating residual connection layer from " << from << "..." << std::endl;
        layer = (new ResidualConnectionLayer(inputSize, sequenceLength, batchSize, scale))
                    ->setFromLayer(layerMap_[from]);
        
    } else if (layerConfig.type == "MapReduce") {
        Logger::log << "Creating MapReduce layer..." << std::endl;
        auto reductionType = layerConfig.params.at("reduction_type").get_value<std::string>();
        layer = new MapReduceLayer(inputSize, outputSize, parseReductionType(reductionType));
        
    } else if (layerConfig.type == "Flatten") {
        Logger::log << "Creating Flatten layer..." << std::endl;
        layer = new FlattenLayer(sequenceLength, inputSize, outputSize, batchSize);
        
    } else if (layerConfig.type == "Reshape") {
        Logger::log << "Creating Reshape layer..." << std::endl;
        layer = new ReshapeLayer(outputSequenceLength, inputSize, outputSize, batchSize);
        
    } else if (layerConfig.type == "PositionalEncoding") {
        Logger::log << "Creating PositionalEncoding layer..." << std::endl;
        assert(outputSequenceLength == sequenceLength);
        layer = new PositionalEncodingLayer(inputSize, sequenceLength, outputSize, batchSize);
    } else {
        throw std::invalid_argument("Unsupported layer type");
    }

    layerMap_[layerName] = layer;
    layer->setIsTerminal(isTerminal);
    layer->setName(layerName);
    layer->buildPipeline(device, library);
    layer->buildBuffers(device);

    return layer;
}
