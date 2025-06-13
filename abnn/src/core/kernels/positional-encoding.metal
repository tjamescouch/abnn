#include <metal_stdlib>

#include "common.metal"

using namespace metal;


kernel void forward_positional_encoding(
    device float* embeddings                 [[buffer(0)]],
    device const float* positional_encodings [[buffer(1)]],
    constant uint& sequence_length           [[buffer(2)]],
    constant uint& embedding_dim             [[buffer(3)]],
    uint gid                                 [[thread_position_in_grid]]
)
{
    embeddings[gid] += positional_encodings[gid % (sequence_length * embedding_dim)];
}

