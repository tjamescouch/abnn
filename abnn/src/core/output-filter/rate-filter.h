#pragma once

#include <vector>
#include <cstddef>   // for std::size_t

/// Continuous-time low-pass filter (optional FIR stage)
class RateFilter {
public:
    /// @param tauSec   time constant τ in seconds (e.g. 0.05)
    /// @param useFIR   whether to apply a trailing moving-average
    /// @param firSize  window length for FIR (if enabled)
    RateFilter(double tauSec,
               bool     useFIR  = true,
               std::size_t firSize = 20)
      : tau_{tauSec}
      , doFIR_{useFIR}
      , firSize_{firSize}
    {}

    /// Process a new raw vector with elapsed dt (seconds).
    /// Returns the filtered (analog) rate.
    std::vector<float> process(const std::vector<float>& raw, double dtSec) {
        const std::size_t N = raw.size();
        if (rate_.empty()) {
            rate_ = raw;                // initialize on first call
        }

        // compute α = dt/(τ + dt)
        const double alpha = dtSec / (tau_ + dtSec);

        // 1) continuous-time low-pass: r += α * (raw − r)
        for (std::size_t i = 0; i < N; ++i) {
            rate_[i] += float(alpha * (raw[i] - rate_[i]));
        }

        // 2) optional FIR smoothing
        if (doFIR_) {
            firHist_.push_back(rate_);
            if (firHist_.size() > firSize_) {
                firHist_.erase(firHist_.begin());
            }

            // compute moving average over the history buffer
            std::vector<float> avg(N, 0.0f);
            for (const auto& frame : firHist_) {
                for (std::size_t i = 0; i < N; ++i) {
                    avg[i] += frame[i];
                }
            }
            const float inv = 1.0f / float(firHist_.size());
            for (auto& v : avg) {
                v *= inv;
            }
            return avg;
        }

        // if no FIR, just return the IIR result
        return rate_;
    }

private:
    double          tau_;       // time‐constant
    bool            doFIR_;     // enable FIR stage?
    std::size_t     firSize_;   // FIR window length

    std::vector<float>                 rate_;     // IIR state
    std::vector<std::vector<float>>    firHist_;  // FIR history
};
