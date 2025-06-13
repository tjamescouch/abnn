#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <vector>
#include <mutex>
#include <functional>
#include <sstream>

namespace MTL {
class Buffer;
}

class Logger {
public:
    static Logger& instance();
    
    void clear();
    
    void logAnalytics(const float* output, int outputCount,
                      const float* target, int targetCount,
                      const uint sequenceLength);
    
    void logMSE(float* targetData, float* outputData, int dimension);
    void logCrossEntropyLoss(float* targetData, float* outputData, int dimension);

    void logLoss(float loss);
    void accumulateLoss(float loss, int currentBatchSize);
    void finalizeBatchLoss();
    
    void addSample(const float* prediction, const float* target);

    void flushAnalytics(const uint sequenceLength);
    
    void clearBatchData();
    void flushBatchData();
    void setBatchSize(int batchSize);
    void setIsRegression(bool isRegression) { isRegression_ = isRegression; }
    
    void printFloatBuffer(MTL::Buffer* b, std::string message);
    void printFloatBuffer(MTL::Buffer* b, std::string message, int maxElements);
    
    void printFloatBufferL2Norm(MTL::Buffer* b, std::string message);
    void printFloatBufferMeanL2Norm(MTL::Buffer* b, std::string message);
    
    void assertBufferContentsAreValid(MTL::Buffer* b, std::string layerName);
    
    void count(MTL::Buffer* b, std::string message, std::function<bool(float)> predicate);
    
    static Logger log;

    template <typename T>
    Logger& operator<<(const T& msg) {
        _stream << msg;
        return *this;
    }

    typedef std::ostream& (*Manipulator)(std::ostream&);
    Logger& operator<<(Manipulator manip) {
        manip(_stream);
        flush();
        return *this;
    }

    
private:
    void flushRegressionAnalytics(const uint sequenceLength);
    void flushClassificationAnalytics();
    
    bool isRegression_ = true;
    float accumulatedLoss_ = 0.0f;
    int numSamples_ = 0;
    int batchSize_ = 1;
    
    std::ofstream *logFileStream = nullptr;
    std::string filename_;
    
    std::vector<std::vector<float>> batchOutputs_;
    std::vector<std::vector<float>> batchTargets_;
    int outputDim_;

    Logger();
    ~Logger();
    static void initSingleton();
    
    static Logger* instance_;
    static std::once_flag initInstanceFlag;
    
    std::stringstream _stream;
    void flush();
};

#endif // LOGGER_H
