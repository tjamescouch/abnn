// functional-dataset.cpp
// ----------------------
// Sliding sinusoid stimulus for ABNN

#include "functional-dataset.h"
#include <cmath>

FunctionalDataset::FunctionalDataset(uint32_t nInput,
                                     double   dtSeconds,
                                     double   freqHz)
: nIn_(nInput)
, dtSec_(dtSeconds)
, freqHz_(freqHz)
{}

std::vector<float> FunctionalDataset::next()
{
    double base = 2.0 * M_PI * freqHz_ * (tick_ * dtSec_);
    ++tick_;

    std::vector<float> v(nIn_);
    for (size_t i = 0; i < nIn_; ++i)
    {
        double phase = base + (2.0 * M_PI * i / nIn_);
        v[i] = 0.5f * (1.0 + std::sin(phase));
    }
    return v;
}

