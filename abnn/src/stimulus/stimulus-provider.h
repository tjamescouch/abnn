//
//  stimulus-provider.h
//  abnn
//
//  Created by James Couch on 2025-06-13.
//

#pragma once
/* stimulus-provider.h  –  abstract interface for input generators
 * ==============================================================
 * Any class that can feed the Brain must inherit from StimulusProvider
 * and implement:
 *
 *   • next()  -> returns a vector<float> of length nInput
 *   • time()  -> returns current stimulus time in seconds
 */

#include <vector>

class StimulusProvider
{
public:
    virtual ~StimulusProvider() = default;

    /* Return the next input frame (length = BrainEngine::nIn_) */
    virtual std::vector<float> nextInput() = 0;
    
    /* Return the next expected output frame (length = BrainEngine::nOut_) */
    virtual std::vector<float> nextExpected() = 0;

    /* Current time of the stimulus in seconds (monotonic) */
    virtual double time() const = 0;
};
