//
//  residual-connection.metal
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include <metal_stdlib>

#include "common.metal"

using namespace metal;

// Adds input and residual connection element-wise.
kernel void forward_residual(
    device const float* input           [[buffer(0)]],
    device const float* residualInput   [[buffer(1)]],
    device float* output                [[buffer(2)]],
    constant float& residualScale       [[buffer(3)]],
    uint gid                            [[thread_position_in_grid]]
) {
    output[gid] = input[gid] + residualScale * residualInput[gid];
}

// Splits gradient backpropagation to input and residual paths.
kernel void backward_residual(
    device const float* inputErrors           [[buffer(0)]],
    device float* outputErrors                [[buffer(1)]],
    device float* residualOutputErrors        [[buffer(2)]],
    constant float& residualScale             [[buffer(3)]],
    uint gid                                  [[thread_position_in_grid]]
) {
    // Propagate gradients equally to input and residual inputs
    outputErrors[gid] += inputErrors[gid];
    residualOutputErrors[gid] += residualScale * inputErrors[gid];
}
