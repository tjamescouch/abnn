// logger.cpp  –  dynamic trace + EMA-loss logger
// ==============================================

#include "logger.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace fs = std::filesystem;

// ---------- ctor ------------------------------------------------------------
Logger::Logger(size_t nIn, size_t nOut,
               size_t windowIn, size_t windowOut,
               float  emaAlpha)
: winIn_(windowIn)
, winOut_(windowOut)
, nIn_(nIn)
, nOut_(nOut)
, emaLoss_(0.f)
, alpha_(emaAlpha)
{
    fs::path p = fs::current_path() / "abnn_session.m";
    matFile_.open(p, std::ios::trunc);
    if (!matFile_)
        std::cerr << "‼️  cannot create " << p << '\n';
    else
        std::cout << "📄  MATLAB log file → " << p << '\n';
}

// ---------- log_samples -----------------------------------------------------
void Logger::log_samples(const std::vector<float>& target,
                         const std::vector<float>& pred)
{
    if (target.size() != nIn_ || pred.size() != nOut_) return;

    std::scoped_lock lock(mtx_);

    if (inBuf_.size()  == winIn_)  inBuf_.pop_front();
    if (outBuf_.size() == winOut_) outBuf_.pop_front();

    inBuf_.push_back(target);
    outBuf_.push_back(pred);

    // EMA MSE loss
    float mse = 0.f;
    for (size_t i=0;i<nOut_;++i)
        mse += (pred[i] - (i<nIn_ ? target[i] : 0.f)) *
               (pred[i] - (i<nIn_ ? target[i] : 0.f));
    mse /= nOut_;
    emaLoss_ = alpha_ * mse + (1.f - alpha_) * emaLoss_;

    std::cout << "\r✨ EMA-Loss: "
              << std::fixed << std::setprecision(6)
              << emaLoss_ << std::flush;
}

// ---------- flush_to_matlab -------------------------------------------------
void Logger::flush_to_matlab()
{
    std::scoped_lock lock(mtx_);
    if (!matFile_) return;
    matFile_.seekp(0, std::ios::beg);

    // channel-0 of input over time
    matFile_ << "input = [";
    for (bool first=true; const auto& v : inBuf_) {
        if (!first) matFile_ << ", "; first=false;
        matFile_ << v[0];
    }
    matFile_ << "];\n";

    // fraction of output neurons that spiked each timestep
    matFile_ << "output = [";
    for (bool first=true; const auto& v : outBuf_) {
        float frac = std::accumulate(v.begin(), v.end(), 0.f) / v.size();
        if (!first) matFile_ << ", "; first=false;
        matFile_ << frac;
    }
    matFile_ << "];\n";

    matFile_ << "figure(1); clf;\n"
                "subplot(2,1,1);\n"
                "plot(input, 'b-'); title('Input channel 0'); ylim([0 1]);\n"
                "subplot(2,1,2);\n"
                "stem(output, 'r.'); title('Fraction of outputs spiked'); "
                "ylim([0 1]);\n"
                "drawnow;\n"
                "disp('Press Ctrl+C to stop viewer'); pause;\n";

    matFile_.flush();
    std::cout << "\n🖋️  MATLAB script updated ("
              << inBuf_.size() << " time-pts)\n";
}
