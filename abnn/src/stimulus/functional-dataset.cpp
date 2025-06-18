#include "functional-dataset.h"
#include <cmath>     /* std::sin, M_PI */
#include "stimulus-provider.h"

/* ctor initialises parameters */
FunctionalDataset::FunctionalDataset(uint32_t nInput,
                                     double   dtSec,
                                     double   freqHz)
: nInput_(nInput)
, dt_     (dtSec)
, fHz_    (freqHz)
, phase_  (0.0)
, tSec_   (0.0)
, v(nInput)
{
}

/* advance phase and return new vector */
std::vector<float> FunctionalDataset::next()
{
    /* advance phase: phase = phase + f * dt  (wrap at 1.0) */
    phase_ += fHz_ * dt_;
    if (phase_ > 1.0) phase_ -= 1.0;
    tSec_  += dt_;

    /* build 0‒1 sine wave across spatial index */

    for (uint32_t i = 0; i < nInput_; ++i) {
        double x = static_cast<double>(i) / nInput_;           /* 0‒1 */
        double s = std::sin(2.0 * M_PI * (x + phase_));        /* -1‒1 */
        v[i] = static_cast<float>(0.5 * (s + 1.0));            /* 0‒1 */
    }
    return v;
}
