#include <metal_stdlib>

#include "common.metal"

using namespace metal;

float activate(const float x, const uint act);
float activate_derivative(const float y, const uint act);




kernel void forward_batch_norm(
    device const float*  input           [[buffer(0)]],
    device float*        output          [[buffer(1)]],
    device const float*  gamma           [[buffer(2)]],
    device const float*  beta            [[buffer(3)]],
    device float*        runningMean     [[buffer(4)]],
    device float*        runningVariance [[buffer(5)]],
    device float*        savedMean       [[buffer(6)]],
    device float*        savedVariance   [[buffer(7)]],
    constant float&      epsilon         [[buffer(8)]],
    constant int&        featureDim      [[buffer(9)]],
    constant bool&       isTraining      [[buffer(10)]],
    constant uint&       batchSize       [[buffer(11)]],
    device float*        debug           [[buffer(12)]],  // not used
    uint                 gid             [[thread_position_in_grid]]
)
{
    if ((int)gid >= featureDim) return;

    // 1) Compute per-batch mean for this feature
    float sum = 0.0f;
    for (uint b = 0; b < batchSize; b++) {
        sum += input[b * featureDim + gid];
    }
    float batchMean = sum / float(batchSize);

    // 2) Compute per-batch variance
    float sqSum = 0.0f;
    for (uint b = 0; b < batchSize; b++) {
        float diff = input[b * featureDim + gid] - batchMean;
        sqSum += diff * diff;
    }
    float batchVar = sqSum / float(batchSize);

    // 3) Save these batch stats for backward pass, no matter what
    savedMean[gid]     = batchMean;
    savedVariance[gid] = batchVar;

    // 4) Update running stats if training
    if (isTraining) {
        // Typically: running = momentum*running + (1 - momentum)*new
        float momentum = 0.9f;
        runningMean[gid]     = momentum * runningMean[gid]     + (1.0f - momentum) * batchMean;
        runningVariance[gid] = momentum * runningVariance[gid] + (1.0f - momentum) * batchVar;
    }

    // 5) Decide which mean and variance to use
    float usedMean     = isTraining ? batchMean  : runningMean[gid];
    float usedVariance = isTraining ? batchVar   : runningVariance[gid];
    float invStd       = rsqrt(usedVariance + epsilon);

    // 6) Normalize each sample’s value
    for (uint b = 0; b < batchSize; b++) {
        float val  = input[b * featureDim + gid];
        float norm = (val - usedMean) * invStd;
        output[b * featureDim + gid] = gamma[gid] * norm + beta[gid];
    }
}



kernel void backward_batch_norm(
    device const float*  input             [[buffer(0)]],
    device const float*  inputErrors       [[buffer(1)]],
    device float*        outputErrors      [[buffer(2)]],
    device const float*  gamma             [[buffer(3)]],
    device const float*  beta              [[buffer(4)]],
    device const float*  savedMean         [[buffer(5)]],
    device const float*  savedVariance     [[buffer(6)]],
    device const float*  runningMean       [[buffer(7)]],
    device const float*  runningVariance   [[buffer(8)]],
    constant float&      epsilon           [[buffer(9)]],
    constant uint&       featureDim        [[buffer(10)]],
    constant bool&       isTraining        [[buffer(11)]],
    constant uint&       batchSize         [[buffer(12)]],
    device float*        debug             [[buffer(13)]],
    device atomic_float* gradientsBeta     [[buffer(14)]],
    device atomic_float* gradientsGamma    [[buffer(15)]],
    uint                 gid               [[thread_position_in_grid]]
)
{
    if (gid >= featureDim) return;

    // Decide mean and variance source
    float mean   = isTraining ? savedMean[gid] : runningMean[gid];
    float var    = isTraining ? savedVariance[gid] : runningVariance[gid];
    float invStd = rsqrt(var + epsilon);

    // Initialize gradient accumulators
    float sum_dY      = 0.0f;
    float sum_dY_xhat = 0.0f;

    // Accumulate gradients for gamma and beta
    for (uint b = 0; b < batchSize; b++) {
        float x = input[b * featureDim + gid];
        float xhat = (x - mean) * invStd;
        float dY = inputErrors[b * featureDim + gid];

        sum_dY      += dY;
        sum_dY_xhat += dY * xhat;
    }

    // Compute averaged gradients
    float grad_gamma = sum_dY_xhat / float(batchSize);
    float grad_beta  = sum_dY / float(batchSize);

    // Write gradients to optimizer buffers atomically
    atomic_fetch_add_explicit(&gradientsGamma[gid], grad_gamma, memory_order_relaxed);
    atomic_fetch_add_explicit(&gradientsBeta[gid], grad_beta, memory_order_relaxed);

    // Compute output errors for previous layer
    for (uint b = 0; b < batchSize; b++) {
        float x = input[b * featureDim + gid];
        float xhat = (x - mean) * invStd;
        float dY = inputErrors[b * featureDim + gid];

        float dX = (gamma[gid] * invStd / float(batchSize)) *
                   (float(batchSize) * dY - sum_dY - xhat * sum_dY_xhat);

        outputErrors[b * featureDim + gid] = dX;
    }
}
