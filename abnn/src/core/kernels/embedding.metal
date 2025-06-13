#include <metal_stdlib>

#include "common.metal"

using namespace metal;


kernel void forward_embedding(
    device const float* token_indices [[buffer(0)]],
    device const float* embeddings [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& embeddingDim [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint token_idx = floor(max(token_indices[gid], 0.f));
    for (uint d = 0; d < embeddingDim; ++d) {
        output[gid * embeddingDim + d] = embeddings[token_idx * embeddingDim + d];
    }
}

kernel void backward_embedding(
    device const float* errors_output [[buffer(0)]],
    device const uint* token_indices [[buffer(1)]],
    device atomic_float* embedding_gradients [[buffer(2)]],
    constant uint& featureDim [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint token_idx = token_indices[gid];
    for (uint d = 0; d < featureDim; ++d) {
        atomic_fetch_add_explicit(&embedding_gradients[token_idx * featureDim + d],
                                  errors_output[gid * featureDim + d],
                                  memory_order_relaxed);
    }
}
