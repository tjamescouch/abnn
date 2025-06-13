//
//  TrainingManager.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-03.
//
#ifndef TRAINING_MANAGER_H
#define TRAINING_MANAGER_H

#include <mutex>

class TrainingManager {
public:
    static TrainingManager& instance();

    bool isTraining() const;
    void setTraining(bool value);

private:
    TrainingManager();
    static void initSingleton();

    static TrainingManager* instance_;
    static std::once_flag initInstanceFlag;

    bool isTraining_;
};


#endif
