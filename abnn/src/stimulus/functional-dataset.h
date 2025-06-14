#pragma once
/* functional-dataset.h  –  phase-shifted sinusoid stimulus
   --------------------------------------------------------
   Implements StimulusProvider so BrainEngine can call next() each pass.   */

#include "brain-engine.h"      // provides StimulusProvider interface
#include <vector>
#include "stimulus-provider.h"

/* Returns a 0‒1 sine wave of length nInput_ whose phase advances dtSec_
   seconds each call (continuous animation). */
class FunctionalDataset : public StimulusProvider
{
public:
    FunctionalDataset(uint32_t nInput,
                      double   dtSec,
                      double   freqHz);

    std::vector<float> next();   /* one frame of stimulus */
    double             time() const { return tSec_; }

private:
    uint32_t nInput_;
    double   dt_;        /* seconds per pass            */
    double   fHz_;       /* sine frequency in Hz        */
    double   phase_;     /* 0‒1 fractional phase        */
    double   tSec_;      /* stimulus time in seconds    */
};
