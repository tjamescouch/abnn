#include <metal_stdlib>

#include "common.metal"

using namespace metal;


float activate(const float x, const uint act);
float activate_derivative(const float y, const uint act);

// Constants defining reduction types for clarity
kernel void forward_map_reduce(
    device const float* input        [[buffer(0)]],
    device float* output             [[buffer(1)]],
    constant int& input_size         [[buffer(2)]],
    constant uint& reductionType     [[buffer(3)]],
    uint tid                         [[thread_position_in_grid]]
) {
    threadgroup float sharedData[1024];

    // Load data into threadgroup memory
    sharedData[tid] = ((int)tid < input_size) ? input[tid] : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction operation
    switch (reductionType) {
        case REDUCTION_SUM:
        case REDUCTION_MEAN:
            for (uint stride = input_size / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    sharedData[tid] += sharedData[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                output[0] = (reductionType == REDUCTION_MEAN) ? (sharedData[0] / input_size) : sharedData[0];
            }
            break;

        case REDUCTION_MAX:
            for (uint stride = input_size / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    sharedData[tid] = max(sharedData[tid], sharedData[tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                output[0] = sharedData[0];
            }
            break;

        case REDUCTION_MIN:
            for (uint stride = input_size / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    sharedData[tid] = min(sharedData[tid], sharedData[tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                output[0] = sharedData[0];
            }
            break;

        case REDUCTION_SOFTMAX:
            if (tid == 0) {
                float maxVal = input[0];
                for (int i = 1; i < input_size; ++i) {
                    maxVal = max(maxVal, input[i]);
                }

                float sumExp = 0.0f;
                for (int i = 0; i < input_size; ++i) {
                    sumExp += exp(input[i] - maxVal);
                }

                for (int i = 0; i < input_size; ++i) {
                    output[i] = exp(input[i] - maxVal) / sumExp;
                }
            }
            break;
    }
}

kernel void backward_map_reduce(
    device const float* outputDelta  [[buffer(0)]],
    device const float* forwardOutput[[buffer(1)]],
    device float* inputErrors        [[buffer(2)]],
    constant uint& input_size        [[buffer(3)]],
    constant uint& reductionType     [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= input_size) return;

    switch (reductionType) {
        case REDUCTION_SUM:
            inputErrors[tid] = outputDelta[0];
            break;

        case REDUCTION_MEAN:
            inputErrors[tid] = outputDelta[0] / input_size;
            break;

        case REDUCTION_MAX:
            inputErrors[tid] = (forwardOutput[tid] == outputDelta[1]) ? outputDelta[0] : 0.0f;
            break;

        case REDUCTION_MIN:
            inputErrors[tid] = (forwardOutput[tid] == outputDelta[1]) ? outputDelta[0] : 0.0f;
            break;

        case REDUCTION_SOFTMAX:
            // forwardOutput is already softmaxed values here
            float grad = 0.0f;
            for (uint i = 0; i < input_size; ++i) {
                float indicator = (tid == i) ? 1.0f : 0.0f;
                grad += outputDelta[i] * forwardOutput[tid] * (indicator - forwardOutput[i]);
            }
            inputErrors[tid] = grad;
            break;
    }
}
