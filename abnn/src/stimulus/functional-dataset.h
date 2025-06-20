#pragma once
/* functional-dataset.h  –  phase-shifted sinusoid stimulus
   --------------------------------------------------------
   Implements StimulusProvider so BrainEngine can call next() each pass.   */

#include "brain-engine.h"      // provides StimulusProvider interface
#include <vector>
#include "stimulus-provider.h"


class FunctionalDataset : public StimulusProvider
{
public:
    FunctionalDataset(uint32_t nInput, uint32_t nOutput, double dtSec, double freqHz, std::function<float(float)> funcInput, std::function<float(float)> funcExpected);

    std::vector<float> nextInput() override;   /* one frame of stimulus */
    std::vector<float> nextExpected() override;   /* one frame of stimulus */
    double             time() const override { return tSec_; }


private:
    uint32_t nInput_;
    uint32_t nOutput_;
    
    double   dt_;        /* seconds per pass            */
    double   tSec_;      /* stimulus time in seconds    */
    double   fHz_;       /* sine frequency in Hz        */
    double   phase_;     /* 0‒1 fractional phase        */
    
    std::function<float(float)> funcInput_;
    std::function<float(float)> funcExpected_;
};
