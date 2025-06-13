#ifndef NEURAL_ENGINE_H
#define NEURAL_ENGINE_H
#include <vector>
#include <functional>

#include "model-config.h"
#include "logger.h"
#include "layer.h"
#include "input-layer.h"
#include "dense-layer.h"
#include "batch-normalization-layer.h"
#include "dataset.h"
#include "data-manager.h"
#include "layer-factory.h"


namespace MTL {
class Device;
class CommandQueue;
class Library;
class CompileOptions;
class CommandBuffer;
class Buffer;
}

class NeuralEngine {
public:
    NeuralEngine(MTL::Device* pDevice, ModelConfig& config, DataManager* pDataManager);
    ~NeuralEngine();
    
    void computeForward(int batchSize, std::function<void()> onComplete);
    void computeBackward(int batchSize, std::function<void()> onComplete);

    void computeForwardBatches(uint32_t totalSamples, int iterationsRemaining, std::function<void()> onComplete);
    void computeBackwardBatches(uint32_t totalSamples, int iterationsRemaining, std::function<void()> onComplete);

    void connectDynamicLayers(ModelConfig& config);
    void createDynamicLayers(ModelConfig& config);
    
    void handleKeyStateChange();
    
    void runTraining();
    void runInference();
    void saveParameters();
    void loadParameters();
    
    void saveModel(const std::string& filepath);
    void loadModel(const std::string& filepath);
    
    
    static constexpr int kMaxFramesInFlight = 3;
    std::vector<Layer*> dynamicLayers_;
    
private:
    void buildComputePipeline();
    
    DataManager* _pDataManager;
    
    InputLayer* _pInputLayer;
    LayerFactory* _pLayerFactory;
    
    MTL::Device* _pDevice;
    MTL::CommandQueue* _pCommandQueue;
    MTL::Library* _pComputeLibrary;
    MTL::CompileOptions* _pCompileOptions;
    
    bool areBuffersBuilt;
    bool currentlyComputing;
    dispatch_semaphore_t _semaphore;
    
    int batch_size;
    int epochs;
    int input_dim;
    int output_dim;
    uint terminalSequenceLength_;
    std::string filename;
};

#endif // NEURAL_ENGINE_H
