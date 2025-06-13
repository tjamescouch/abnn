//
//  configuration-manager.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-05.
//
#ifndef CONFIGURATION_MANAGER_H
#define CONFIGURATION_MANAGER_H

#include <mutex>
#include "model-config.h"

class ConfigurationManager {
public:
    static ConfigurationManager& instance();

    ModelConfig* getConfig() const;
    void setConfig(ModelConfig *pConfig);

private:
    ConfigurationManager();
    static void initSingleton();

    static ConfigurationManager* instance_;
    static std::once_flag initInstanceFlag;

    ModelConfig* _pConfig;
};


#endif
