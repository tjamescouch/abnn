//
//  TrainingManager.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-03.
//
#include "training-manager.h"



TrainingManager* TrainingManager::instance_ = nullptr;
std::once_flag TrainingManager::initInstanceFlag;

TrainingManager::TrainingManager() : isTraining_(true) {}

TrainingManager& TrainingManager::instance() {
    std::call_once(initInstanceFlag, &TrainingManager::initSingleton);
    return *instance_;
}

bool TrainingManager::isTraining() const { return isTraining_; }

void TrainingManager::setTraining(bool value) { isTraining_ = value; }

void TrainingManager::initSingleton() {
    instance_ = new TrainingManager();
}

