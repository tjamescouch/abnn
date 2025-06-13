#include <metal_stdlib>

#include "common.metal"

using namespace metal;

#define MAX_MODEL_DIM 1024

inline uint i2D(uint width, uint row, uint col) {
    return row * width + col;
}

kernel void forward_self_attention(
    device const float* input                [[buffer(0)]],  // [batchSize, seqLength, inputDim]
    device const float* weightsQ             [[buffer(1)]],  // [inputDim, modelDim]
    device const float* weightsK             [[buffer(2)]],  // [inputDim, modelDim]
    device const float* weightsV             [[buffer(3)]],  // [inputDim, modelDim]
    device const float* weightsO             [[buffer(4)]],  // [modelDim, inputDim]

    device float* bufferQ                    [[buffer(5)]],  // [batchSize, seqLength, modelDim]
    device float* bufferK                    [[buffer(6)]],  // [batchSize, seqLength, modelDim]
    device float* bufferV                    [[buffer(7)]],  // [batchSize, seqLength, modelDim]


    device float* output                     [[buffer(8)]],  // [batchSize, seqLength, inputDim]

    constant uint& batchSize                 [[buffer(9)]],
    constant uint& seqLength                 [[buffer(10)]],
    constant uint& inputDim                  [[buffer(11)]],
    constant uint& modelDim                  [[buffer(12)]],

    uint gid                                 [[thread_position_in_grid]])
{
    if (gid >= batchSize * seqLength) return;

    uint batchIdx = gid / seqLength;
    uint seqIdx = gid % seqLength;

    // Offset calculations
    uint input_offset = (batchIdx * seqLength + seqIdx) * inputDim;
    uint buffer_offset = (batchIdx * seqLength + seqIdx) * modelDim;
    uint output_offset = input_offset;

    // Step 1: Compute Q, K, V vectors
    for (uint m = 0; m < modelDim; ++m) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        float v_sum = 0.0f;

        for (uint i = 0; i < inputDim; ++i) {
            float in_val = input[input_offset + i];
            q_sum += in_val * weightsQ[i * modelDim + m];
            k_sum += in_val * weightsK[i * modelDim + m];
            v_sum += in_val * weightsV[i * modelDim + m];
        }
        bufferQ[buffer_offset + m] = q_sum;
        bufferK[buffer_offset + m] = k_sum;
        bufferV[buffer_offset + m] = v_sum;
    }

    threadgroup float attention_scores[512]; // Adjust size if necessary

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute Attention Scores (Scaled Dot-Product)
    float scale = rsqrt(float(modelDim));

    for (uint seqIdx2 = 0; seqIdx2 < seqLength; ++seqIdx2) {
        float attn_score = 0.0f;
        for (uint m = 0; m < modelDim; ++m) {
            float q_val = bufferQ[(batchIdx * seqLength + seqIdx) * modelDim + m];
            float k_val = bufferK[(batchIdx * seqLength + seqIdx2) * modelDim + m];
            attn_score += q_val * k_val;
        }
        attn_score *= scale;
        attention_scores[seqIdx2] = attn_score;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Softmax on attention scores
    float max_score = attention_scores[0];
    for (uint i = 1; i < seqLength; ++i)
        max_score = max(max_score, attention_scores[i]);

    float sum_exp = 0.0f;
    for (uint seqIdx2 = 0; seqIdx2 < seqLength; ++seqIdx2) {
        attention_scores[seqIdx2] = exp(attention_scores[seqIdx2] - max_score);
        sum_exp += attention_scores[seqIdx2];
    }

    for (uint seqIdx2 = 0; seqIdx2 < seqLength; ++seqIdx2) {
        attention_scores[seqIdx2] /= sum_exp;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Weighted sum of V vectors
    float context[MAX_MODEL_DIM]; // allocate context array on stack
    for (uint m = 0; m < modelDim; ++m) {
        float context_sum = 0.0f;
        for (uint seqIdx2 = 0; seqIdx2 < seqLength; ++seqIdx2) {
            float attn = attention_scores[seqIdx2];
            float v_val = bufferV[(batchIdx * seqLength + seqIdx2) * modelDim + m];
            context_sum += attn * v_val;
        }
        context[m] = context_sum;
    }

    // Step 4b: Write context back to original space using weightsO
    for (uint i = 0; i < inputDim; ++i) {
        float out_sum = 0.0f;
        for (uint m = 0; m < modelDim; ++m) {
            out_sum += context[m] * weightsO[m * inputDim + i];
        }
        output[output_offset + i] = out_sum;
    }
}


// Tune for your GPU
#define THREADGROUP_SIZE 1024
#define TILE_DIM         16

// Utility atomic add (Metal 3+ usually has atomic_float natively)
inline void atomicAdd(volatile device atomic_float* ptr, float val)
{
    atomic_fetch_add_explicit(ptr, val, memory_order_relaxed);
}


kernel void backward_self_attention(
    device const float* input                [[buffer(0)]],
    device const float* weightsQ             [[buffer(1)]],
    device const float* weightsK             [[buffer(2)]],
    device const float* weightsV             [[buffer(3)]],
    device const float* weightsO             [[buffer(4)]],

    device const float* bufferQ              [[buffer(5)]],
    device const float* bufferK              [[buffer(6)]],
    device const float* bufferV              [[buffer(7)]],
    device const float* attn_weights         [[buffer(8)]],

    device atomic_float* inputErrors         [[buffer(9)]],
    device const float* outputErrors         [[buffer(10)]],

    device atomic_float* gradWeightsQ        [[buffer(11)]],
    device atomic_float* gradWeightsK        [[buffer(12)]],
    device atomic_float* gradWeightsV        [[buffer(13)]],
    device atomic_float* gradWeightsO        [[buffer(14)]],

    constant uint& batchSize                 [[buffer(15)]],
    constant uint& seqLength                 [[buffer(16)]],
    constant uint& inputDim                  [[buffer(17)]],
    constant uint& modelDim                  [[buffer(18)]],

    // Large scratch buffer, sized for batchSize*seqLength threads
    device float* scratch                    [[buffer(19)]],

    uint tid                                 [[thread_position_in_threadgroup]],
    uint blockId                             [[threadgroup_position_in_grid]],
    uint threadsPerGroup                     [[threads_per_threadgroup]],
    uint gridSize                            [[threads_per_grid]]
)
{
    // Global thread index
    uint gid = blockId * threadsPerGroup + tid;
    uint totalTokens = batchSize * seqLength;
    if (gid >= totalTokens) return;

    // Map to (b, s)
    uint b = gid / seqLength;
    uint s = gid % seqLength;

    // Offsets
    uint inputOffset  = (b * seqLength + s) * inputDim;
    uint outErrOffset = (b * seqLength + s) * inputDim;
    uint attnOff      = (b * seqLength * seqLength) + (s * seqLength);

    //--------------------------------------------
    // PART 0: define scratch layout per thread
    //--------------------------------------------
    // We'll store:
    //   - dAttn[modelDim]
    //   - attnVal[modelDim]
    //   - dAttnW_raw[seqLength]
    //   - dAttnW[seqLength]
    //   - dV_s[seqLength * modelDim]
    //   - dK_s[seqLength * modelDim]
    //   - dQ[modelDim]
    //
    // We'll also store dOut_i[inputDim] and a small inVal[inputDim] in scratch
    // instead of private arrays to avoid 'hard-coded 256'.
    //
    // So total needed = 3*modelDim + 2*seqLength + 2*(seqLength*modelDim) + modelDim + inputDim + inputDim
    //                 = 4*modelDim + 2*seqLength + 2*(seqLength * modelDim) + 2*inputDim.
    //--------------------------------------------
    uint scratchPerThread = (4*modelDim) + (2*seqLength) + (2*seqLength*modelDim) + (2*inputDim);

    uint base = gid * scratchPerThread;
    device float* dAttn_d    = scratch + base;                           // [modelDim]
    device float* attnVal_d  = dAttn_d + modelDim;                       // [modelDim]
    device float* dAttnW_raw = attnVal_d + modelDim;                     // [seqLength]
    device float* dAttnW     = dAttnW_raw + seqLength;                   // [seqLength]
    device float* dV_s       = dAttnW + seqLength;                       // [seqLength*modelDim]
    device float* dK_s       = dV_s + (seqLength*modelDim);              // [seqLength*modelDim]
    device float* dQ         = dK_s + (seqLength*modelDim);              // [modelDim]
    device float* dOut_i     = dQ + modelDim;                            // [inputDim]
    device float* inVal      = dOut_i + inputDim;                        // [inputDim]

    //--------------------------------------------
    // PART 1: read dOut from global => dOut_i
    //--------------------------------------------
    for (uint i = 0; i < inputDim; i++) {
        dOut_i[i] = outputErrors[outErrOffset + i];
    }

    //--------------------------------------------
    // PART 2: dAttn = dOut * weightsO^T
    //--------------------------------------------
    for (uint d = 0; d < modelDim; d++) {
        float sumVal = 0.0f;
        for (uint i = 0; i < inputDim; i++) {
            sumVal += dOut_i[i] * weightsO[d*inputDim + i];
        }
        dAttn_d[d] = sumVal;
    }

    //--------------------------------------------
    // PART 3: attnVal_d = sum_j( attn_weights(b,s,j)* V(b,j) )
    //--------------------------------------------
    for (uint d = 0; d < modelDim; d++) {
        float sumVal = 0.0f;
        for (uint j = 0; j < seqLength; j++) {
            float aw = attn_weights[attnOff + j];
            uint vOff = (b*seqLength + j)*modelDim + d;
            sumVal += aw * bufferV[vOff];
        }
        attnVal_d[d] = sumVal;
    }

    //--------------------------------------------
    // PART 4: gradWeightsO += outer( attnVal_d, dOut_i )
    //         tile partial sums in threadgroup
    //--------------------------------------------
    threadgroup float partialGradO[TILE_DIM*TILE_DIM];
    for (uint dStart = 0; dStart < modelDim; dStart += TILE_DIM) {
        for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
            // zero tile
            for (uint idx = tid; idx < TILE_DIM*TILE_DIM; idx += threadsPerGroup) {
                partialGradO[idx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // accumulate
            for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                uint ld = localIdx / TILE_DIM;
                uint li = localIdx % TILE_DIM;
                uint d_ = dStart + ld;
                uint i_ = iStart + li;
                if (d_ < modelDim && i_ < inputDim) {
                    float val = attnVal_d[d_] * dOut_i[i_];
                    partialGradO[localIdx] = val;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // atomic add
            for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                uint ld = localIdx / TILE_DIM;
                uint li = localIdx % TILE_DIM;
                uint d_ = dStart + ld;
                uint i_ = iStart + li;
                if (d_ < modelDim && i_ < inputDim) {
                    atomicAdd(&(gradWeightsO[d_*inputDim + i_]), partialGradO[localIdx]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    //--------------------------------------------
    // PART 5: dV_s => for j in [0..seqLength], dV_s(j,d) = attnW(b,s,j)* dAttn(d)
    //--------------------------------------------
    for (uint j = 0; j < seqLength; j++) {
        float aw = attn_weights[attnOff + j];
        for (uint d = 0; d < modelDim; d++) {
            dV_s[j*modelDim + d] = aw * dAttn_d[d];
        }
    }

    //--------------------------------------------
    // PART 6: dAttnW_raw => dot(dAttn, V(b,j)), then apply softmax derivative
    //--------------------------------------------
    float sumSoftmax = 0.0f;
    // fill dAttnW_raw
    for (uint j = 0; j < seqLength; j++) {
        float accum = 0.0f;
        uint vOff_j = (b*seqLength + j)*modelDim;
        for (uint d = 0; d < modelDim; d++) {
            accum += dAttn_d[d] * bufferV[vOff_j + d];
        }
        dAttnW_raw[j] = accum;
    }
    // sum for softmax
    for (uint j = 0; j < seqLength; j++) {
        float aw = attn_weights[attnOff + j];
        sumSoftmax += aw * dAttnW_raw[j];
    }
    // final
    for (uint j = 0; j < seqLength; j++) {
        float aw = attn_weights[attnOff + j];
        dAttnW[j] = aw * (dAttnW_raw[j] - sumSoftmax);
    }

    //--------------------------------------------
    // PART 7: dQ, dK => from dAttnW
    //--------------------------------------------
    float scale = 1.0f / sqrt((float)modelDim);
    // zero dQ
    for (uint d = 0; d < modelDim; d++) {
        dQ[d] = 0.0f;
    }
    // zero dK_s
    for (uint j = 0; j < seqLength; j++) {
        for (uint d = 0; d < modelDim; d++) {
            dK_s[j*modelDim + d] = 0.0f;
        }
    }
    // accumulate
    uint qOff_bs = (b*seqLength + s)*modelDim;
    for (uint j = 0; j < seqLength; j++) {
        float coef = dAttnW[j] * scale;
        // dQ
        uint kOff_j = (b*seqLength + j)*modelDim;
        for (uint d = 0; d < modelDim; d++) {
            dQ[d] += coef * bufferK[kOff_j + d];
        }
        // dK_s
        for (uint d = 0; d < modelDim; d++) {
            dK_s[j*modelDim + d] = coef * bufferQ[qOff_bs + d];
        }
    }

    //--------------------------------------------
    // PART 8: Q => accumulate inputErrors(b,s)
    //            + gradWeightsQ
    //--------------------------------------------
    // read input(b,s) => inVal
    for (uint i = 0; i < inputDim; i++) {
        inVal[i] = input[inputOffset + i];
    }

    // dInput(b,s,i) = sum_d( dQ[d]*weightsQ[i,d] )
    // We'll do a partial sum in shared memory to reduce collisions, if desired

    // For demonstration, let's tile partial sums for inputErrors too:
    threadgroup float partialInputQ[TILE_DIM];
    for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
        // zero tile
        for (uint idx = tid; idx < TILE_DIM; idx += threadsPerGroup) {
            partialInputQ[idx] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // accumulate
        for (uint localIdx = tid; localIdx < TILE_DIM; localIdx += threadsPerGroup) {
            uint i_ = iStart + localIdx;
            if (i_ < inputDim) {
                float accum = 0.0f;
                for (uint d = 0; d < modelDim; d++) {
                    accum += dQ[d] * weightsQ[i_*modelDim + d];
                }
                partialInputQ[localIdx] = accum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // atomic add
        for (uint localIdx = tid; localIdx < TILE_DIM; localIdx += threadsPerGroup) {
            uint i_ = iStart + localIdx;
            if (i_ < inputDim) {
                float val = partialInputQ[localIdx];
                atomicAdd(&(inputErrors[inputOffset + i_]), val);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Now gradWeightsQ => outer(inVal, dQ)
    threadgroup float partialGradQ[TILE_DIM*TILE_DIM];
    for (uint dStart = 0; dStart < modelDim; dStart += TILE_DIM) {
        for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
            // zero tile
            for (uint idx = tid; idx < TILE_DIM*TILE_DIM; idx += threadsPerGroup) {
                partialGradQ[idx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // accumulate
            for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                uint ld = localIdx / TILE_DIM;
                uint li = localIdx % TILE_DIM;
                uint dd = dStart + ld;
                uint ii = iStart + li;
                if (dd < modelDim && ii < inputDim) {
                    float val = inVal[ii] * dQ[dd];
                    partialGradQ[localIdx] = val;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // atomic add
            for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                uint ld = localIdx / TILE_DIM;
                uint li = localIdx % TILE_DIM;
                uint dd = dStart + ld;
                uint ii = iStart + li;
                if (dd < modelDim && ii < inputDim) {
                    atomicAdd(&(gradWeightsQ[ii*modelDim + dd]), partialGradQ[localIdx]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    //--------------------------------------------
    // PART 9: K => accumulate inputErrors(b,j) + gradWeightsK
    //--------------------------------------------
    // same approach: for each j in [seqLength]
    for (uint j = 0; j < seqLength; j++) {
        uint inOff_j = (b*seqLength + j)*inputDim;
        // read input(b,j) => inVal
        for (uint i = 0; i < inputDim; i++) {
            inVal[i] = input[inOff_j + i];
        }

        // dInput(b,j,i) from dK_s[j,d]*weightsK[i,d] => tile partial sums
        for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
            for (uint idx = tid; idx < TILE_DIM; idx += threadsPerGroup) {
                partialInputQ[idx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint localIdx = tid; localIdx < TILE_DIM; localIdx += threadsPerGroup) {
                uint i_ = iStart + localIdx;
                if (i_ < inputDim) {
                    float accum = 0.0f;
                    for (uint d = 0; d < modelDim; d++) {
                        accum += dK_s[j*modelDim + d] * weightsK[i_*modelDim + d];
                    }
                    partialInputQ[localIdx] = accum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint localIdx = tid; localIdx < TILE_DIM; localIdx += threadsPerGroup) {
                uint i_ = iStart + localIdx;
                if (i_ < inputDim) {
                    float val = partialInputQ[localIdx];
                    atomicAdd(&(inputErrors[inOff_j + i_]), val);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // gradWeightsK => outer(inVal, dK_s[j])
        for (uint dStart = 0; dStart < modelDim; dStart += TILE_DIM) {
            for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
                for (uint idx = tid; idx < TILE_DIM*TILE_DIM; idx += threadsPerGroup) {
                    partialGradQ[idx] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                    uint ld = localIdx / TILE_DIM;
                    uint li = localIdx % TILE_DIM;
                    uint dd = dStart + ld;
                    uint ii = iStart + li;
                    if (dd < modelDim && ii < inputDim) {
                        float val = inVal[ii] * dK_s[j*modelDim + dd];
                        partialGradQ[localIdx] = val;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                    uint ld = localIdx / TILE_DIM;
                    uint li = localIdx % TILE_DIM;
                    uint dd = dStart + ld;
                    uint ii = iStart + li;
                    if (dd < modelDim && ii < inputDim) {
                        atomicAdd(&(gradWeightsK[ii*modelDim + dd]), partialGradQ[localIdx]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    //--------------------------------------------
    // PART 10: V => accumulate inputErrors + gradWeightsV
    //--------------------------------------------
    for (uint j = 0; j < seqLength; j++) {
        uint inOff_j = (b*seqLength + j)*inputDim;
        // read input(b,j) => inVal
        for (uint i = 0; i < inputDim; i++) {
            inVal[i] = input[inOff_j + i];
        }

        // dInput(b,j,i)
        for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
            for (uint idx = tid; idx < TILE_DIM; idx += threadsPerGroup) {
                partialInputQ[idx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint localIdx = tid; localIdx < TILE_DIM; localIdx += threadsPerGroup) {
                uint i_ = iStart + localIdx;
                if (i_ < inputDim) {
                    float accum = 0.0f;
                    for (uint d = 0; d < modelDim; d++) {
                        accum += dV_s[j*modelDim + d] * weightsV[i_*modelDim + d];
                    }
                    partialInputQ[localIdx] = accum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint localIdx = tid; localIdx < TILE_DIM; localIdx += threadsPerGroup) {
                uint i_ = iStart + localIdx;
                if (i_ < inputDim) {
                    atomicAdd(&(inputErrors[inOff_j + i_]), partialInputQ[localIdx]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // gradWeightsV => outer(inVal, dV_s[j])
        for (uint dStart = 0; dStart < modelDim; dStart += TILE_DIM) {
            for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
                for (uint idx = tid; idx < TILE_DIM*TILE_DIM; idx += threadsPerGroup) {
                    partialGradQ[idx] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                    uint ld = localIdx / TILE_DIM;
                    uint li = localIdx % TILE_DIM;
                    uint dd = dStart + ld;
                    uint ii = iStart + li;
                    if (dd < modelDim && ii < inputDim) {
                        float val = inVal[ii] * dV_s[j*modelDim + dd];
                        partialGradQ[localIdx] = val;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                    uint ld = localIdx / TILE_DIM;
                    uint li = localIdx % TILE_DIM;
                    uint dd = dStart + ld;
                    uint ii = iStart + li;
                    if (dd < modelDim && ii < inputDim) {
                        atomicAdd(&(gradWeightsV[ii*modelDim + dd]), partialGradQ[localIdx]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    // Done.
}
