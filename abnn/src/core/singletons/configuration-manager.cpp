//
//  configuration-manager.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-05.
//
#include "configuration-manager.h"

ConfigurationManager* ConfigurationManager::instance_ = nullptr;
std::once_flag ConfigurationManager::initInstanceFlag;

ConfigurationManager::ConfigurationManager() : _pConfig(nullptr) {}

ConfigurationManager& ConfigurationManager::instance() {
    std::call_once(initInstanceFlag, &ConfigurationManager::initSingleton);
    return *instance_;
}

ModelConfig* ConfigurationManager::getConfig() const { return _pConfig; }

void ConfigurationManager::setConfig(ModelConfig* pConfig) { _pConfig = pConfig; }

void ConfigurationManager::initSingleton() {
    instance_ = new ConfigurationManager();
}

