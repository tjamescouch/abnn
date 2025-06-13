#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "layer.h"
#include <vector>

// Forward declarations for Metal classes.
namespace MTL {
    class Device;
    class Buffer;
}
class Layer;

class InputLayer : public Layer {
public:
    InputLayer(int sequenceLength, int inputDim, int batchSize);
    ~InputLayer();
    
    void buildBuffers(MTL::Device* device) override;
    void updateBufferAt(const float*);
    void updateBufferAt(const float*, int batchSize);
    void buildPipeline(MTL::Device* device, MTL::Library* library) override {};
    void forward(MTL::CommandBuffer* cmdBuf, int batchSize) override {};
    void backward(MTL::CommandBuffer* cmdBuf, int batchSize) override {};
    
    void resetErrors() override;

    int inputSize() const override { return inputDim_; }
    int outputSize() const override { return inputDim_; }
    void updateTargetBufferAt(const float* targetData) override {};
    void updateTargetBufferAt(const float* targetData, int batchSize) override {};

    void setInputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBuffer(BufferType type) override;

    void setOutputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBuffer(BufferType type) override;
    
    void connectForwardConnections(Layer* previousLayer) override {};
    void connectBackwardConnections(Layer* previousLayer) override {};
    
    
    void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    
    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;
    
    void setIsTerminal(bool isTerminal) override { isTerminal_ = isTerminal; };
    
    void debugLog() override {
#ifdef DEBUG_INPUT_LAYER
        for (int t = 0; t < sequenceLength_; t++) {
            float* outputs = static_cast<float*>(outputBuffers_[BufferType::Output][t]->contents());
            printf("[InputLayer Output Debug] timestep %d: ", t);
            for(int i = 0; i < outputBuffers_[BufferType::Output][t]->length()/sizeof(float); ++i)
                printf(" %f, ", outputs[i]);
            printf("\n");
        }
#endif
    }
    
private:
    int inputDim_;
    bool isTerminal_;
    int batchSize_;
    int sequenceLength_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
};

#endif // INPUT_LAYER_H
