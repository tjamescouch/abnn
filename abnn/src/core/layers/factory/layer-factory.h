//
//  layer-factory.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-04.
//

#ifndef LAYER_FACTORY_H
#define LAYER_FACTORY_H

#include "layer.h"
#include "model-config.h"
#include <Metal/Metal.hpp>
#include <unordered_map>

class LayerFactory {
public:
    LayerFactory();
    Layer* createLayer(LayerConfig& layerConfig,
                              MTL::Device* device,
                              MTL::Library* library,
                              bool isTerminal);
    
private:
    std::unordered_map<std::string, Layer*> layerMap_;
    int layerIdCounter_ = 0;  // Incrementing ID starting from 0

};

#endif // LAYER_FACTORY_H
