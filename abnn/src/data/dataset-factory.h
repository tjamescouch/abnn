//
//  dataset-factory.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-21.
//

#pragma once

#include "dataset.h"
#include "model-config.h"

// Explicitly responsible for dataset creation based on ModelConfig
class DatasetFactory {
public:
    // Static method explicitly returns dataset instances
    static Dataset* createDataset(const ModelConfig* _pConfig);
};
