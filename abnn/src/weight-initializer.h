//
//  weight-initializer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#pragma once
#include <random>

class WeightInitializer {
public:
    static void initializeXavier(float* buffer, int inputDim, int outputDim);
    static void initializeHe(float* buffer, int inputDim, int outputDim);
    static void initializeBias(float* buffer, int dim, float scale = 0.01f);
    static void initializeZeros(float* buffer, int dim);
};
