//
//  weight-initializer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#include "weight-initializer.h"
#include <cassert>

void WeightInitializer::initializeXavier(float* buffer, int inputDim, int outputDim) {
    float xavier_scale = sqrtf(6.0f / (inputDim + outputDim));
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-xavier_scale, xavier_scale);
    for (int i = 0; i < inputDim * outputDim; ++i) {
        buffer[i] = dist(rng);
        assert(!isnan(buffer[i]));
    }
}

void WeightInitializer::initializeHe(float* buffer, int inputDim, int outputDim) {
    float he_scale = sqrtf(2.0f / inputDim);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> dist(0.0f, he_scale);

    for (int i = 0; i < inputDim * outputDim; ++i) {
        buffer[i] = dist(rng);
        assert(!isnan(buffer[i]));
    }
}

void WeightInitializer::initializeBias(float* buffer, int dim, float scale) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (int i = 0; i < dim; ++i) {
        buffer[i] = dist(rng);
        assert(!isnan(buffer[i]));
    }
}

void WeightInitializer::initializeZeros(float* buffer, int dim) {
    for (int i = 0; i < dim; ++i) {
        buffer[i] = 0.0f;
    }
}
