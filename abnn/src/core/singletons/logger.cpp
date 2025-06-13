#include "logger.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include "math-lib.h"
#include "app-kit-bridge.h"

const char* filename = "multilayer_nn_training.m";

Logger::Logger()
:
filename_(filename),
logFileStream(nullptr),
isRegression_(true),
batchSize_(1) {
    logFileStream = new std::ofstream(filename_, std::ios::app);
    if (!logFileStream->is_open()) {
        std::cerr << "Error opening log file: " << filename_ << std::endl;
    }
}

Logger::~Logger() {
    if (logFileStream) {
        if (logFileStream->is_open())
            logFileStream->close();
        delete logFileStream;
        logFileStream = nullptr;
    }
}

Logger* Logger::instance_ = nullptr;
std::once_flag Logger::initInstanceFlag;

void Logger::flushAnalytics(const uint sequenceLength) {
    if (isRegression_) {
        return flushRegressionAnalytics(sequenceLength);
    }
    
    return flushClassificationAnalytics();
}

void Logger::logAnalytics(const float* output, int outputCount,
                          const float* target, int targetCount,
                          const uint sequenceLength) {
    assert(outputCount % sequenceLength == 0 && "Output count must be divisible by sequence length.");
    assert(targetCount % sequenceLength == 0 && "Target count must be divisible by sequence length.");

    batchOutputs_.emplace_back(output, output + outputCount);
    batchTargets_.emplace_back(target, target + targetCount);
}

void Logger::flushRegressionAnalytics(const uint sequenceLength) {
    if (!logFileStream->is_open()) {
        std::cerr << "Error opening log file: " << filename_ << std::endl;
        return;
    }

    size_t numSamples = batchOutputs_.size();
    size_t outputDim = batchTargets_[0].size() / sequenceLength;

    for (size_t sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx) {
        const auto& output = batchOutputs_[sampleIdx];
        const auto& target = batchTargets_[sampleIdx];

        *logFileStream << "clf; hold on;" << std::endl;
        *logFileStream << "ylim([-1 1], \"Manual\");" << std::endl;

        for (size_t seqIdx = 0; seqIdx < sequenceLength; ++seqIdx) {
            *logFileStream << "target = [";
            for (size_t i = 0; i < outputDim; ++i) {
                size_t idx = seqIdx * outputDim + i;
                *logFileStream << target[idx] << (i < outputDim - 1 ? ", " : "");
            }
            *logFileStream << "];" << std::endl;

            *logFileStream << "output = [";
            for (size_t i = 0; i < outputDim; ++i) {
                size_t idx = seqIdx * outputDim + i;
                *logFileStream << output[idx] << (i < outputDim - 1 ? ", " : "");
            }
            *logFileStream << "];" << std::endl;

            *logFileStream << "scatter(1:" << outputDim << ", target, 'filled', 'b', 'DisplayName', 'Target Seq " << seqIdx+1 << "');" << std::endl;
            *logFileStream << "scatter(1:" << outputDim << ", output, 'filled', 'r', 'DisplayName', 'Prediction Seq " << seqIdx+1 << "');" << std::endl;
        }

        *logFileStream << "legend('show');" << std::endl;
        *logFileStream << "pause(0.01);" << std::endl;
    }

    batchOutputs_.clear();
    batchTargets_.clear();
}

void Logger::flushClassificationAnalytics() {
    if (!logFileStream->is_open()) {
        std::cerr << "Error opening log file: " << filename_ << std::endl;
        return;
    }


    size_t numClasses = batchOutputs_[0].size();
    
    
    if (numClasses == 0) {
        Logger::log << "Error: numClasses is zero, invalid logger state." << std::endl;
        return;
    }

    for (size_t sampleIdx = 0; sampleIdx < batchOutputs_.size(); ++sampleIdx) {
        const auto& output = batchOutputs_[sampleIdx];
        const auto& target = batchTargets_[sampleIdx];
        
        size_t offset = 0; // explicitly first sample

        *logFileStream << "clf; hold on;" << std::endl;
        *logFileStream << "xlabel('Class (Digit)'); ylabel('Probability');" << std::endl;
        *logFileStream << "ylim([0, 1]);" << std::endl;
        *logFileStream << "x = 0:" << (numClasses - 1) << ";" << std::endl;

        // Target vector explicitly using offset
        *logFileStream << "target = [";
        for (int i = 0; i < numClasses; ++i) {
            *logFileStream << target[offset + i] << (i < numClasses - 1 ? ", " : "") << " ";
        }
        *logFileStream << "];" << std::endl;

        // Output probabilities explicitly using offset
        *logFileStream << "output = [";
        for (int i = 0; i < numClasses; ++i) {
            *logFileStream << output[offset + i] << (i < numClasses - 1 ? ", " : "") << " ";
        }
        *logFileStream << "];" << std::endl;

        // Plot commands remain unchanged
        *logFileStream << "bar(x - 0.15, target, 0.3, 'FaceColor', 'b', 'DisplayName', 'Target');" << std::endl;
        *logFileStream << "bar(x + 0.15, output, 0.3, 'FaceColor', 'r', 'DisplayName', 'Prediction');" << std::endl;
        *logFileStream << "legend('show');" << std::endl;
        *logFileStream << "pause(0.05);" << std::endl;
    }

    *logFileStream << "hold off;" << std::endl;
}

void Logger::clear() {
    // Close the existing member stream if open.
    if (logFileStream && logFileStream->is_open()) {
        logFileStream->close();
    }
    
    {
        std::ofstream ofs(filename_, std::ios::trunc);
        if (!ofs.is_open()) {
            std::cerr << "Error clearing log file: " << filename_ << std::endl;
            return;
        }
    }
    
    logFileStream->open(filename_, std::ios::app);
    if (!logFileStream->is_open()) {
        std::cerr << "Error reopening log file: " << filename_ << std::endl;
    }
}

void Logger::logLoss(float loss) {
    Logger::log << "✨ Loss: " << loss << std::endl;
}

void Logger::accumulateLoss(float loss, int batchSize) {
    accumulatedLoss_ += loss;
    numSamples_+= batchSize;
    logLoss(accumulatedLoss_ / numSamples_);
}

void Logger::finalizeBatchLoss() {
    accumulatedLoss_ = 0.0f;
    numSamples_ = 0;
}

void Logger::clearBatchData() {
    batchOutputs_.clear();
    batchTargets_.clear();
}

void Logger::setBatchSize(int batchSize) {
    batchSize_ = batchSize;
}

Logger& Logger::instance() {
    std::call_once(initInstanceFlag, &Logger::initSingleton);
    return *instance_;
}

void Logger::initSingleton() {
    instance_ = new Logger();
}

void Logger::assertBufferContentsAreValid(MTL::Buffer* b, std::string layerName) {
    float* data = static_cast<float*>(b->contents());
    size_t numFloats = b->length() / sizeof(float);
    bool nonZeroFound = false;
    bool largeValuesFound = false;
    bool assertionsSatisfied = true;
    float maxval = 100;
    
    for (int i = 0; i < numFloats; ++i) {
        if(isnan(data[i])) {
            log << "Error in layer " << layerName << " : buffer contains nan" << std::endl;
            assertionsSatisfied = false;
            
            break;
        }
        if(isinf(data[i])) {
            log << "Error in layer " << layerName << " : buffer contains inf" << std::endl;
            assertionsSatisfied = false;
            
            break;
        }
        
        float absval = abs(data[i]);
        
        if (absval > 0) {
            nonZeroFound = true;
        }
        
        if (absval > maxval) {
            largeValuesFound = true;
        }
    }
    
    if (!nonZeroFound) {
        //log << "Warning in layer " << layerName << " : buffer is all 0" << std::endl;
    }
    
    if (largeValuesFound) {
        //log << "Warning in layer " << layerName << " : large values found" << std::endl;
    }
    
    if(!assertionsSatisfied) {
        printFloatBuffer(b, "Dumping buffer: ");
        //assert(false);
    }
}

void Logger::printFloatBuffer(MTL::Buffer* b, std::string message, int maxElements) {
    float* data = static_cast<float*>(b->contents());
    size_t numFloats = mathlib::min<size_t>(b->length() / sizeof(float), maxElements);
    
    Logger::log << message << " => [";
    for (int i = 0; i < numFloats; ++i) {
        Logger::log << data[i];
        if (i < numFloats - 1) {
            Logger::log << ", ";
        }
    }
    Logger::log << "]" << std::endl;
}

void Logger::count(MTL::Buffer* b, std::string message, std::function<bool(float)> predicate) {
    float* data = static_cast<float*>(b->contents());
    size_t numFloats = b->length() / sizeof(float);
    
    size_t count = 0;
    for (int i = 0; i < numFloats; ++i) {
        if (predicate(data[i])) {
            count++;
        }
    }
    Logger::log << message << " => " << count << std::endl;
}

void Logger::printFloatBuffer(MTL::Buffer* b, std::string message) {
    this->printFloatBuffer(b, message, 500);
}

void Logger::printFloatBufferL2Norm(MTL::Buffer* b, std::string message) {
    float* data = static_cast<float*>(b->contents());
    size_t numFloats = b->length() / sizeof(float);

    float norm = 0.0f;
    for (size_t i = 0; i < numFloats; ++i)
        norm += data[i] * data[i];
    
    norm = sqrtf(norm);
    Logger::log << message << " => " << norm << std::endl;
}

void Logger::printFloatBufferMeanL2Norm(MTL::Buffer* b, std::string message) {
    float* data = static_cast<float*>(b->contents());
    size_t numFloats = b->length() / sizeof(float);

    float norm = 0.0f;
    for (size_t i = 0; i < numFloats; ++i)
        norm += data[i] * data[i];
    
    norm = sqrtf(norm) / numFloats;
    Logger::log << message << " => " << norm << std::endl;
}

Logger Logger::log; // Static instance initialization

void Logger::flush() {
    std::string output = _stream.str();
    updateTextField(output.c_str());  // Your existing Objective-C bridge
    std::cout << output.c_str();
    _stream.str(std::string()); // clear buffer after flush
    _stream.clear();
}
