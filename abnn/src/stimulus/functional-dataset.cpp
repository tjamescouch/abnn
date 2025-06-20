#include "functional-dataset.h"
#include <cmath>     /* std::sin, M_PI */
#include "stimulus-provider.h"

/* ctor initialises parameters */
FunctionalDataset::FunctionalDataset(uint32_t nInput,
                                     uint32_t nOutput,
                                     double dtSec,
                                     double freqHz,
                                     std::function<float(float)> funcInput,
                                     std::function<float(float)> funcExpected)
: nInput_ (nInput)
, nOutput_(nOutput)
, funcInput_(funcInput)
, funcExpected_(funcExpected)
, dt_     (dtSec)
, fHz_    (freqHz)
, phase_  (0.0)
, tSec_   (0.0)
{
}

/* advance phase and return new inputvector */
std::vector<float> FunctionalDataset::nextInput()
{
    std::vector<float> v(nInput_);
    
    /* advance phase: phase = phase + f * dt  (wrap at 1.0) */
    phase_ += fHz_ * dt_;
    if (phase_ > 1.0) phase_ -= 1.0;
    tSec_  += dt_;

    for (uint32_t i = 0; i < nInput_; ++i) {
        double x = static_cast<double>(i) / nInput_;           /* 0‒1 */
        float s = funcInput_((float)(2.0 * M_PI * (x + phase_)));
        v[i] = s;//static_cast<float>(0.5 * (s + 1.0));            /* 0‒1 */
    }
    return v;
}


std::vector<float> FunctionalDataset::nextExpected()
{
    std::vector<float> v(nOutput_);

    for (uint32_t i = 0; i < nOutput_; ++i) {
        double x = static_cast<double>(i) / nOutput_;           /* 0‒1 */
        double s = funcExpected_(2.0 * M_PI * (x + phase_));
        v[i] = s;//static_cast<float>(0.5 * (s + 1.0));            /* 0‒1 */
    }
    return v;
}
