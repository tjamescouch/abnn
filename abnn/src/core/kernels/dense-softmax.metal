#include <metal_stdlib>
using namespace metal;

#include "common.metal"

float activate(const float x, const uint act);
float activate_derivative(const float y, const uint act);

// Clamping thresholds
constant float threshold    = 1.f;

kernel void forward_softmax_dense_layer(
                                device const float* h         [[buffer(0)]],
                                device float* y               [[buffer(1)]],
                                device const float* W         [[buffer(2)]],
                                device const float* b         [[buffer(3)]],
                                constant uint& hidden_dim     [[buffer(4)]],
                                constant uint& output_dim     [[buffer(5)]],
                                constant uint& activation     [[buffer(6)]],
                                constant uint& batchSize      [[buffer(7)]],
                                device float* debug           [[buffer(8)]],
                                uint gid                      [[thread_position_in_grid]]
                                )
{
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;

    if (sample_id >= batchSize || neuron_id >= output_dim) return;

    // Compute linear sum (dense layer)
    float sum = b[neuron_id];
    for (uint i = 0; i < hidden_dim; ++i) {
        sum += h[sample_id * hidden_dim + i] * W[i * output_dim + neuron_id];
    }

    // Store logits temporarily in global memory
    y[sample_id * output_dim + neuron_id] = sum;
    
    threadgroup_barrier(mem_flags::mem_device);

    // Compute softmax
    if (neuron_id == 0) {
        float maxLogit = y[sample_id * output_dim];
        for (uint i = 1; i < output_dim; i++) {
            maxLogit = max(maxLogit, y[sample_id * output_dim + i]);
        }

        float sumExp = 0.0f;
        for (uint i = 0; i < output_dim; i++) {
            y[sample_id * output_dim + i] = exp(y[sample_id * output_dim + i] - maxLogit);
            sumExp += y[sample_id * output_dim + i];
        }

        float denominator = max(sumExp, 1e-9f);
        for (uint i = 0; i < output_dim; i++) {
            y[sample_id * output_dim + i] /= denominator;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Write the normalized softmax outputs to debug for verification
    debug[sample_id * output_dim + neuron_id] = y[sample_id * output_dim + neuron_id];
}

kernel void backward_non_terminal_softmax_dense_layer(
                                           device const float* h                [[buffer(0)]],  // input activations
                                           device float*       W                [[buffer(1)]],  // weights
                                           device const float* b                [[buffer(2)]],  // biases
                                           device const float* y_hat            [[buffer(3)]],  // layer outputs
                                           device const float* inputErrors      [[buffer(4)]],  // errors fed INTO this layer from next
                                           device float*       outputError      [[buffer(5)]],  // errors fed BACK to previous (delta)
                                           constant uint&      input_dim        [[buffer(6)]],
                                           constant uint&      output_dim       [[buffer(7)]],
                                           constant uint&      activation       [[buffer(8)]],
                                           device float*       prevLayerErrors  [[buffer(9)]], // final error to the previous layer's activations
                                           constant uint&      batch_size       [[buffer(10)]],
                                           device atomic_float* gradientsW      [[buffer(11)]],
                                           device atomic_float* gradientsB      [[buffer(12)]],
                                           uint tid                               [[thread_position_in_threadgroup]],
                                           uint gid                               [[thread_position_in_grid]]
                                           )
{
    // Identify sample + neuron
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;
    
    if (sample_id >= batch_size || neuron_id >= output_dim) return;
    
    // Pointers to this sample's portion of each buffer
    const device float* sample_h         = h + (sample_id * input_dim);
    const device float* sample_y_hat     = y_hat + (sample_id * output_dim);
    const device float* sample_inErrors  = inputErrors + (sample_id * output_dim);
    device float*       sample_outError  = outputError + (sample_id * output_dim);
    device float*       sample_prevError = prevLayerErrors + (sample_id * input_dim);
    
    // raw_error is the incoming gradient wrt this neuron's output
    float raw_error = sample_inErrors[neuron_id];
    
    // Delta = error * activation derivative
    float dAct = activate_derivative(sample_y_hat[neuron_id], activation);
    float delta = raw_error * dAct;
    
    // Clamp to avoid numerical blow-up
    delta = clamp(delta, -threshold, threshold);
    
    // Save partial delta in outputError (i.e., error from this layer's viewpoint)
    sample_outError[neuron_id] = delta;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint i = 0; i < input_dim; i++) {
        float grad = sample_h[i] * delta;  // computed gradient (partial dW)
        
        // Write explicitly to gradients buffer instead of modifying W
        uint gradWIdx = i * output_dim + neuron_id;
        atomic_fetch_add_explicit(&gradientsW[gradWIdx], grad, memory_order_relaxed);
        
        // Use existing weights (unchanged) to propagate errors backward
        float weightVal = W[i * output_dim + neuron_id];
        float prevErrTerm = weightVal * delta;
        
        atomic_fetch_add_explicit((device atomic_float*)&sample_prevError[i],
                                  prevErrTerm, memory_order_relaxed);
    }
    
    // Update bias gradient explicitly (no direct bias updates)
    device atomic_float* pGradientsB = (device atomic_float*)&gradientsB[neuron_id];
    atomic_fetch_add_explicit(pGradientsB, delta, memory_order_relaxed);
}

// You may tune this constant to trade off register usage vs. fewer atomic operations.
constant uint CHUNK_SIZE = 128;

// This kernel is a drop-in replacement for a terminal dense layer with a softmax output.
// We assume cross-entropy loss, so the backprop delta for each output neuron is:
//   delta_j = (pred_j - target_j)
kernel void backward_terminal_softmax_dense_layer(
    device const float*      inputs             [[buffer(0)]],  // [batch_size * input_dim]
    device const float*      weights            [[buffer(1)]],  // [input_dim * output_dim]
    device const float*      predictions        [[buffer(2)]],  // [batch_size * output_dim]
    device const float*      targets            [[buffer(3)]],  // [batch_size * output_dim]
    device atomic_float*     gradientWeights    [[buffer(4)]],  // [input_dim * output_dim]
    device atomic_float*     gradientBiases     [[buffer(5)]],  // [output_dim]
    device float*            errorsPreviousLayer [[buffer(6)]], // [batch_size * input_dim]
    constant uint&           input_dim          [[buffer(7)]],
    constant uint&           output_dim         [[buffer(8)]],
    constant uint&           batch_size         [[buffer(9)]],
    // Even though softmax doesn't need an activation ID, we keep the signature consistent
    // in case your framework expects it:
    uint2 gid                                 [[thread_position_in_grid]],
    uint2 tid                                 [[thread_position_in_threadgroup]]
)
{
    // Each thread handles a single (input_idx, sample_idx) pair
    uint input_idx  = gid.x;
    uint sample_idx = gid.y;

    // Bounds check
    if (input_idx >= input_dim || sample_idx >= batch_size) {
        return;
    }

    // Load the input activation for this thread’s (sample, input)
    float inputVal = inputs[sample_idx * input_dim + input_idx];

    // We'll accumulate partial gradients in chunks to reduce the number of atomic operations.
    // We'll also accumulate the error to backprop to the previous layer in a running variable.
    float errorLocal = 0.0f;

    // Loop over output neurons in chunks of CHUNK_SIZE
    for (uint chunkStart = 0; chunkStart < output_dim; chunkStart += CHUNK_SIZE)
    {
        uint chunkEnd = min(chunkStart + CHUNK_SIZE, output_dim);
        uint chunkLen = chunkEnd - chunkStart;

        // Private arrays to store partial sums for dW and dB
        float gradWLocal[CHUNK_SIZE];
        float gradBLocal[CHUNK_SIZE];

        // Initialize accumulators for this chunk
        for (uint i = 0; i < chunkLen; i++) {
            gradWLocal[i] = 0.0f;
            gradBLocal[i] = 0.0f;
        }

        // Accumulate partial gradients within the chunk
        for (uint out = chunkStart; out < chunkEnd; out++)
        {
            // Compute index within the chunk
            uint localIdx = out - chunkStart;

            // Retrieve predictions and targets
            float pred   = predictions[sample_idx * output_dim + out];
            float target = targets[sample_idx * output_dim + out];

            // For softmax + cross-entropy, delta = (y_hat - y)
            float delta = (pred - target);

            // Accumulate gradient wrt. weight = inputVal * delta
            gradWLocal[localIdx] += (inputVal * delta);

            // Accumulate gradient wrt. bias = delta
            gradBLocal[localIdx] += delta;

            // Accumulate error for backprop to previous layer
            float w = weights[input_idx * output_dim + out];
            errorLocal += (w * delta);
        }

        // Write partial sums to global memory using atomic adds
        for (uint i = 0; i < chunkLen; i++)
        {
            uint outIdx = chunkStart + i;

            // Gradient wrt. weight
            uint globalWIdx = (input_idx * output_dim) + outIdx;
            atomic_fetch_add_explicit(&gradientWeights[globalWIdx],
                                      gradWLocal[i],
                                      memory_order_relaxed);

            // Gradient wrt. bias
            atomic_fetch_add_explicit(&gradientBiases[outIdx],
                                      gradBLocal[i],
                                      memory_order_relaxed);
        }
    }

    // Write the accumulated error to propagate back to the previous layer
    errorsPreviousLayer[sample_idx * input_dim + input_idx] = errorLocal;
}
