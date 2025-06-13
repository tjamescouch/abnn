#pragma once
// logger.h  –  dynamic-length trace logger for ABNN
// -----------------------------------------------------------------
// • Works with any N_INPUT / N_OUTPUT sent at construction time.
// • Keeps up to windowIn_ / windowOut_ most recent samples.
// • log_samples(targetVec, predictionVec)  ← whole vectors
// • flush_to_matlab() writes/overwrites "abnn_session.m"
//   which plots the rolling traces and then pauses so the
//   window stays open.
//
// Compile as C++17.
// -----------------------------------------------------------------

#include <deque>
#include <vector>
#include <fstream>
#include <mutex>

class Logger
{
public:
    Logger(size_t nIn,
           size_t nOut,
           size_t windowIn  = 256,
           size_t windowOut = 128,
           float  emaAlpha  = 0.01f);

    // copy-disabled
    Logger(const Logger&)            = delete;
    Logger& operator=(const Logger&) = delete;

    // push one full sample vector
    void log_samples(const std::vector<float>& target,
                     const std::vector<float>& prediction);

    // write/refresh abnn_session.m
    void flush_to_matlab();

    // current EMA loss
    float ema_loss() const { return emaLoss_; }

private:
    // ring buffers
    std::deque<std::vector<float>> inBuf_;
    std::deque<std::vector<float>> outBuf_;
    size_t  winIn_, winOut_;

    // dims
    size_t  nIn_, nOut_;

    // loss
    float emaLoss_;
    float alpha_;

    // file + mutex
    std::ofstream matFile_;
    std::mutex    mtx_;
};
