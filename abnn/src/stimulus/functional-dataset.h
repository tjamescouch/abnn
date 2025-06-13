#pragma once
// functional-dataset.h  –  StimulusProvider producing a sliding sinusoid
// -----------------------------------------------------------------------------
// • Constructor parameters
//       nInput    : number of input neurons (will be the vector length)
//       dtSeconds : virtual time advanced per Brain pass
//       freqHz    : sine-wave frequency (default 1 Hz)
//
// • next() returns std::vector<float>(nInput) with values in [0,1].
//
// Channel 0 is 0.5*(1+sin(2π f t)).
// Other channels copy channel 0 by default (edit .cpp to customise).
// -----------------------------------------------------------------------------

#include "brain-engine.h"
#include <vector>
#include <functional>

class FunctionalDataset : public StimulusProvider
{
public:
    FunctionalDataset(uint32_t nInput,
                      double   dtSeconds,
                      double   freqHz = 1.0);

    std::vector<float> next() override;
    double             time() const override { return tick_ * dtSec_; }
    void               rewind() { tick_ = 0; }

private:
    uint32_t nIn_;
    double   dtSec_;
    double   freqHz_;
    uint64_t tick_ = 0;     // sample index
};
