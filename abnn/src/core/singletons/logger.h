#pragma once
#include <vector>
#include <fstream>
#include <string>

class Logger
{
public:
    Logger(int nInput, int nOutput);
    ~Logger();

    /* animated frame */
    void log_samples(const std::vector<float>& input,
                     const std::vector<float>& output);

    /* loss tracking */
    void accumulate_loss(double loss);
    void flush_loss();             /* prints EMA */

private:
    int  nIn_, nOut_;
    std::ofstream mat_;

    /* loss EMA */
    double ema_ = 0.0;
    double beta_= 0.98;            /* smoothing */
    uint64_t step_ = 0;
};
