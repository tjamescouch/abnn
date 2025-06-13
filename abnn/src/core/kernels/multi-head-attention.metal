#include <metal_stdlib>

#include "common.metal"

using namespace metal;

#define MAX_SEQ_LENGTH 512
#define MAX_MODEL_DIM 1024

inline uint i2D(uint width, uint row, uint col) {
    return row * width + col;
}


kernel void forward_multi_head_attention(
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
    constant uint& numHeads                  [[buffer(13)]],
    constant float& scale                    [[buffer(14)]],

    uint gid                                 [[thread_position_in_grid]])
{
    if (gid >= batchSize * seqLength) return;

    uint batchIdx = gid / seqLength;
    uint seqIdx = gid % seqLength;

    uint input_offset = (batchIdx * seqLength + seqIdx) * inputDim;
    uint buffer_offset = (batchIdx * seqLength + seqIdx) * modelDim;

    uint headDim = modelDim / numHeads;

    // Compute Q, K, V for each head explicitly
    for (uint head = 0; head < numHeads; ++head) {
        uint head_offset = head * headDim;

        // Compute Q, K, V for each head separately
        for (uint m = 0; m < headDim; ++m) {
            float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
            for (uint i = 0; i < inputDim; ++i) {
                float in_val = input[input_offset + i];
                q_sum += in_val * weightsQ[i2D(modelDim, i, head_offset + m)];
                k_sum += in_val * weightsK[i2D(modelDim, i, head_offset + m)];
                v_sum += in_val * weightsV[i2D(modelDim, i, head_offset + m)];
            }
            bufferQ[buffer_offset + head_offset + m] = q_sum;
            bufferK[buffer_offset + head_offset + m] = k_sum;
            bufferV[buffer_offset + head_offset + m] = v_sum;
        }

        threadgroup float attn_scores[MAX_SEQ_LENGTH];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute Attention scores
        for (uint s = 0; s < seqLength; ++s) {
            float score = 0.0f;
            for (uint m = 0; m < headDim; ++m) {
                float q_val = bufferQ[buffer_offset + head_offset + m];
                float k_val = bufferK[(batchIdx * seqLength + s) * modelDim + head_offset + m];
                score += q_val * k_val;
            }
            attn_scores[s] = score * scale;
        }

        // Softmax
        float max_attn = attn_scores[0];
        for (uint s = 1; s < seqLength; ++s)
            max_attn = max(max_attn, attn_scores[s]);

        float sum_exp = 0.0f;
        for (uint s = 0; s < seqLength; ++s) {
            attn_scores[s] = exp(attn_scores[s] - max_attn);
            sum_exp += attn_scores[s];
        }

        for (uint s = 0; s < seqLength; ++s)
            attn_scores[s] /= sum_exp;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Weighted sum (context vector)
        for (uint m = 0; m < headDim; ++m) {
            float context = 0.0f;
            for (uint s = 0; s < seqLength; ++s) {
                float attn = attn_scores[s];
                float v_val = bufferV[(batchIdx * seqLength + s) * modelDim + head_offset + m];
                context += attn * v_val;
            }
            bufferQ[buffer_offset + head_offset + m] = context;
        }
    }

    // Project concatenated heads back to inputDim
    for (uint i = 0; i < inputDim; ++i) {
        float out_sum = 0.0f;
        for (uint m = 0; m < modelDim; ++m) {
            out_sum += bufferQ[buffer_offset + m] * weightsO[m * inputDim + i];
        }
        output[input_offset + i] = out_sum;
    }
}


// For your tuning
#define TILE_DIM 16
#define THREADGROUP_SIZE 256


// Atomic add utility (Metal 3+ typically has native atomic_float)
inline void atomicAdd(volatile device atomic_float* ptr, float val)
{
    atomic_fetch_add_explicit(ptr, val, memory_order_relaxed);
}

kernel void backward_multi_head_attention(
    device const float*  input         [[buffer(0)]],   // [batchSize * seqLength * inputDim]
    device const float*  weightsQ      [[buffer(1)]],   // [inputDim * modelDim]
    device const float*  weightsK      [[buffer(2)]],   // [inputDim * modelDim]
    device const float*  weightsV      [[buffer(3)]],   // [inputDim * modelDim]
    device const float*  weightsO      [[buffer(4)]],   // [modelDim * inputDim]

    device const float*  bufferQ       [[buffer(5)]],   // [batchSize * seqLength * modelDim]
    device const float*  bufferK       [[buffer(6)]],
    device const float*  bufferV       [[buffer(7)]],
    device const float*  attn_weights  [[buffer(8)]],   // [batchSize * numHeads * seqLength * seqLength]

    device atomic_float* inputErrors   [[buffer(9)]],   // [batchSize * seqLength * inputDim]
    device const float*  outputErrors  [[buffer(10)]],  // [batchSize * seqLength * inputDim]

    device atomic_float* gradWeightsQ  [[buffer(11)]],  // [inputDim * modelDim]
    device atomic_float* gradWeightsK  [[buffer(12)]],
    device atomic_float* gradWeightsV  [[buffer(13)]],
    device atomic_float* gradWeightsO  [[buffer(14)]],

    constant uint& batchSize           [[buffer(15)]],
    constant uint& seqLength           [[buffer(16)]],
    constant uint& inputDim            [[buffer(17)]],
    constant uint& modelDim            [[buffer(18)]],

    // Large scratch buffer for big arrays
    device float*       scratch        [[buffer(19)]],

    constant uint&      numHeads       [[buffer(20)]],
                                          
    constant float& scale              [[buffer(21)]],

    // Thread info
    uint tid                           [[thread_position_in_threadgroup]],
    uint blockId                       [[threadgroup_position_in_grid]],
    uint threadsPerGroup               [[threads_per_threadgroup]],
    uint gridSize                      [[threads_per_grid]]
)
{
    // 1) Global thread ID => (b, s)
    uint gid = blockId * threadsPerGroup + tid;
    uint totalTokens = batchSize * seqLength;
    if (gid >= totalTokens) return;

    uint b = gid / seqLength;
    uint s = gid % seqLength;

    // Offsets to input & output
    uint inputOffset  = (b * seqLength + s)*inputDim;
    uint outErrOffset = (b * seqLength + s)*inputDim;

    // Each thread handles one (b,s).
    // We'll do a loop over heads => sub-dimension headDim = modelDim / numHeads
    uint headDim = modelDim / numHeads;

    //---------------------------------------------
    // 2) We define how much scratch each thread needs
    //    same breakdown as earlier
    //---------------------------------------------
    uint scratchPerThread = (2*inputDim) +    // dOut_i, inVal
                            (3*headDim)    +  // dAttn_head, attnVal_head, dQ_head
                            (2*seqLength*headDim) + // dV_s, dK_s
                            (2*seqLength);    // dAttnW_raw, dAttnW

    // offset in scratch for this thread
    uint base = gid * scratchPerThread;

    // partition the scratch space
    device float* dOut_i       = scratch + base;
    device float* inVal        = dOut_i + inputDim;
    device float* dAttn_head   = inVal + inputDim;
    device float* attnVal_head = dAttn_head + headDim;
    device float* dQ_head      = attnVal_head + headDim;
    device float* dV_s         = dQ_head + headDim;            // size= seqLength*headDim
    device float* dK_s         = dV_s + (seqLength*headDim);   // size= seqLength*headDim
    device float* dAttnW_raw   = dK_s + (seqLength*headDim);   // size= seqLength
    device float* dAttnW       = dAttnW_raw + seqLength;       // size= seqLength

    //---------------------------------------------
    // 3) Copy the dOut for this token => dOut_i
    //---------------------------------------------
    for (uint i = 0; i < inputDim; i++) {
        dOut_i[i] = outputErrors[outErrOffset + i];
    }

    //---------------------------------------------
    // We'll define a small threadgroup array for partial sums:
    //---------------------------------------------
    threadgroup float partialTile[TILE_DIM*TILE_DIM];
    // Also for partial sums for input errors, a 1D tile
    threadgroup float partial1D[TILE_DIM];

    // We'll do the same partial-sum approach for each of the big accumulations
    // (gradWeightsO, gradWeightsQ/K/V, inputErrors).

    //---------------------------------------------
    // 4) Loop over heads. For each head:
    //    (a) compute dAttn_head, attnVal_head
    //    (b) gradWeightsO (tiled)
    //    (c) dV_s, dAttnW_raw => softmax derivative => dAttnW
    //    (d) dQ_head, dK_s
    //    (e) tile-based accumulations for inputErrors + gradWeightsQ/K/V
    //---------------------------------------------
    for (uint h = 0; h < numHeads; h++)
    {
        uint dOff = h * headDim;  // slice in [0..headDim) of modelDim

        //-----------------------------------------
        // A) dAttn_head = dOut_i * (weightsO slice)^T
        //    row= (dOff + d), col= i => index= (dOff + d)*inputDim + i
        //-----------------------------------------
        for (uint d = 0; d < headDim; d++) {
            float sumVal = 0.0f;
            uint rowW = dOff + d;
            for (uint i = 0; i < inputDim; i++) {
                sumVal += dOut_i[i] * weightsO[rowW*inputDim + i];
            }
            dAttn_head[d] = sumVal;
        }

        //-----------------------------------------
        // B) attnVal_head[d] = sum_j( attnW(b,h,s,j)* V(b,h,j, dOff+d) )
        //    attnOffset => b*(numHeads*seqLength*seqLength) + h*(seqLength*seqLength) + s*seqLength
        //-----------------------------------------
        uint attnBase = b*(numHeads*seqLength*seqLength) + h*(seqLength*seqLength) + s*(seqLength);

        for (uint d = 0; d < headDim; d++) {
            float sumVal = 0.0f;
            for (uint j = 0; j < seqLength; j++) {
                float aw = attn_weights[attnBase + j];
                uint vOff = (b*seqLength + j)*modelDim + (dOff + d);
                sumVal += aw * bufferV[vOff];
            }
            attnVal_head[d] = sumVal;
        }

        //-----------------------------------------
        // C) gradWeightsO => outer(attnVal_head, dOut_i)
        //    shape= [headDim, inputDim], offset in [dOff..dOff+headDim) for rows
        //-----------------------------------------
        for (uint dStart = 0; dStart < headDim; dStart += TILE_DIM) {
            for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
                // zero tile
                for (uint idx = tid; idx < TILE_DIM*TILE_DIM; idx += threadsPerGroup) {
                    partialTile[idx] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // accumulate partial sums
                for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                    uint ld = localIdx / TILE_DIM;   // row offset in tile
                    uint li = localIdx % TILE_DIM;   // col offset in tile
                    uint d_ = dStart + ld;
                    uint i_ = iStart + li;
                    if (d_ < headDim && i_ < inputDim) {
                        float val = attnVal_head[d_] * dOut_i[i_];
                        partialTile[localIdx] = val;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // atomic add to gradWeightsO
                for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                    uint ld = localIdx / TILE_DIM;
                    uint li = localIdx % TILE_DIM;
                    uint d_ = dStart + ld;
                    uint i_ = iStart + li;
                    if (d_ < headDim && i_ < inputDim) {
                        // actual row in O = dOff + d_
                        uint rowO = (dOff + d_);
                        float val = partialTile[localIdx];
                        atomicAdd(&(gradWeightsO[rowO*inputDim + i_]), val);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        //-----------------------------------------
        // D) dV_s => for each j => attnW(b,h,s,j) * dAttn_head
        //-----------------------------------------
        for (uint j = 0; j < seqLength; j++) {
            float aw = attn_weights[attnBase + j];
            for (uint d = 0; d < headDim; d++) {
                dV_s[j*headDim + d] = aw * dAttn_head[d];
            }
        }

        //-----------------------------------------
        // E) dAttnW_raw => dot(dAttn_head, V(b,h,j)), then softmax derivative => dAttnW
        //-----------------------------------------
        for (uint j = 0; j < seqLength; j++) {
            float sumVal = 0.0f;
            uint vOff_j = (b*seqLength + j)*modelDim + dOff;
            for (uint d = 0; d < headDim; d++) {
                sumVal += dAttn_head[d] * bufferV[vOff_j + d];
            }
            dAttnW_raw[j] = sumVal;
        }
        float sumSoftmax = 0.0f;
        for (uint j = 0; j < seqLength; j++) {
            float aw = attn_weights[attnBase + j];
            sumSoftmax += aw * dAttnW_raw[j];
        }
        for (uint j = 0; j < seqLength; j++) {
            float aw = attn_weights[attnBase + j];
            dAttnW[j] = aw * (dAttnW_raw[j] - sumSoftmax);
        }

        //-----------------------------------------
        // F) dQ_head, dK_s => from dAttnW
        //    scale= 1 / sqrt(headDim)
        //-----------------------------------------
        
        // zero dQ_head
        for (uint d = 0; d < headDim; d++) {
            dQ_head[d] = 0.0f;
        }
        // zero dK_s
        for (uint j = 0; j < seqLength; j++) {
            for (uint d = 0; d < headDim; d++) {
                dK_s[j*headDim + d] = 0.0f;
            }
        }
        // accumulate
        uint qOff_bs = (b*seqLength + s)*modelDim + dOff;
        for (uint j = 0; j < seqLength; j++) {
            float coef = dAttnW[j] * scale;
            uint kOff_j = (b*seqLength + j)*modelDim + dOff;
            // dQ_head
            for (uint d = 0; d < headDim; d++) {
                dQ_head[d] += coef * bufferK[kOff_j + d];
            }
            // dK_s
            for (uint d = 0; d < headDim; d++) {
                dK_s[j*headDim + d] = coef * bufferQ[qOff_bs + d];
            }
        }

        //-----------------------------------------
        // G) Accumulate into inputErrors + gradWeightsQ/K/V with TILING
        //-----------------------------------------

        //-----------------------------------------
        // G1) Q => read input(b,s) => inVal
        //    then tile partial sums for inputErrors and gradWeightsQ
        //-----------------------------------------
        for (uint i = 0; i < inputDim; i++) {
            inVal[i] = input[inputOffset + i];
        }

        // inputErrors(b,s) => we do a 1D tile approach for partial sums
        // dInput(b,s,i) = sum_d( dQ_head[d]* weightsQ[i, (dOff+d)] )
        for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
            // zero partial1D
            for (uint idx = tid; idx < TILE_DIM; idx += threadsPerGroup) {
                partial1D[idx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint localIdx = tid; localIdx < TILE_DIM; localIdx += threadsPerGroup) {
                uint i_ = iStart + localIdx;
                if (i_ < inputDim) {
                    float accum = 0.0f;
                    for (uint d = 0; d < headDim; d++) {
                        uint idxW = i_*modelDim + (dOff + d);
                        accum += dQ_head[d]* weightsQ[idxW];
                    }
                    partial1D[localIdx] = accum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // atomicAdd to inputErrors
            for (uint localIdx = tid; localIdx < TILE_DIM; localIdx += threadsPerGroup) {
                uint i_ = iStart + localIdx;
                if (i_ < inputDim) {
                    atomicAdd(&(inputErrors[inputOffset + i_]), partial1D[localIdx]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // gradWeightsQ => shape= [inputDim, modelDim], subrange rows= i in [0..inputDim], cols= dOff..dOff+headDim
        // tile-based partial sum
        for (uint dStart = 0; dStart < headDim; dStart += TILE_DIM) {
            for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
                // zero partialTile
                for (uint idx = tid; idx < TILE_DIM*TILE_DIM; idx += threadsPerGroup) {
                    partialTile[idx] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // accumulate partial
                for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                    uint ld = localIdx / TILE_DIM;
                    uint li = localIdx % TILE_DIM;
                    uint d_ = dStart + ld;
                    uint i_ = iStart + li;
                    if (d_ < headDim && i_ < inputDim) {
                        float val = inVal[i_] * dQ_head[d_];
                        partialTile[localIdx] = val;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // atomic add
                for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                    uint ld = localIdx / TILE_DIM;
                    uint li = localIdx % TILE_DIM;
                    uint d_ = dStart + ld;
                    uint i_ = iStart + li;
                    if (d_ < headDim && i_ < inputDim) {
                        uint col = (dOff + d_);
                        atomicAdd(&(gradWeightsQ[i_*modelDim + col]), partialTile[localIdx]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        //-----------------------------------------
        // G2) K => for each j => tile partial sums for input(b,j) & gradWeightsK
        //-----------------------------------------
        for (uint j_ = 0; j_ < seqLength; j_++) {
            uint inOff_j = (b*seqLength + j_)*inputDim;
            // read input(b,j_) => inVal
            for (uint i = 0; i < inputDim; i++) {
                inVal[i] = input[inOff_j + i];
            }

            // dInput(b,j_,i) = sum_d( dK_s[j_,d]* weightsK[i,(dOff+d)] )
            for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
                for (uint idx = tid; idx < TILE_DIM; idx += threadsPerGroup) {
                    partial1D[idx] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint localIdx = tid; localIdx < TILE_DIM; localIdx += threadsPerGroup) {
                    uint i_ = iStart + localIdx;
                    if (i_ < inputDim) {
                        float accum = 0.0f;
                        for (uint d = 0; d < headDim; d++) {
                            uint idxW = i_*modelDim + (dOff + d);
                            accum += dK_s[j_*headDim + d]* weightsK[idxW];
                        }
                        partial1D[localIdx] = accum;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint localIdx = tid; localIdx < TILE_DIM; localIdx += threadsPerGroup) {
                    uint i_ = iStart + localIdx;
                    if (i_ < inputDim) {
                        atomicAdd(&(inputErrors[inOff_j + i_]), partial1D[localIdx]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // gradWeightsK => outer(inVal, dK_s[j_]) in tiles
            for (uint dStart = 0; dStart < headDim; dStart += TILE_DIM) {
                for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
                    for (uint idx = tid; idx < TILE_DIM*TILE_DIM; idx += threadsPerGroup) {
                        partialTile[idx] = 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                        uint ld = localIdx / TILE_DIM;
                        uint li = localIdx % TILE_DIM;
                        uint d_ = dStart + ld;
                        uint i_ = iStart + li;
                        if (d_ < headDim && i_ < inputDim) {
                            float val = inVal[i_] * dK_s[j_*headDim + d_];
                            partialTile[localIdx] = val;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                        uint ld = localIdx / TILE_DIM;
                        uint li = localIdx % TILE_DIM;
                        uint d_ = dStart + ld;
                        uint i_ = iStart + li;
                        if (d_ < headDim && i_ < inputDim) {
                            uint col = (dOff + d_);
                            atomicAdd(&(gradWeightsK[i_*modelDim + col]), partialTile[localIdx]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        } // end for j_ in [seqLength]

        //-----------------------------------------
        // G3) V => from dV_s
        //-----------------------------------------
        for (uint j_ = 0; j_ < seqLength; j_++) {
            uint inOff_j = (b*seqLength + j_)*inputDim;
            // read input(b,j_) => inVal
            for (uint i = 0; i < inputDim; i++) {
                inVal[i] = input[inOff_j + i];
            }

            // dInput(b,j_,i) from sum_d( dV_s[j_, d]* weightsV[i, (dOff+d)] )
            for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
                for (uint idx = tid; idx < TILE_DIM; idx += threadsPerGroup) {
                    partial1D[idx] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint localIdx = tid; localIdx < TILE_DIM; localIdx += threadsPerGroup) {
                    uint i_ = iStart + localIdx;
                    if (i_ < inputDim) {
                        float accum = 0.0f;
                        for (uint d = 0; d < headDim; d++) {
                            uint idxW = i_*modelDim + (dOff + d);
                            accum += dV_s[j_*headDim + d]* weightsV[idxW];
                        }
                        partial1D[localIdx] = accum;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint localIdx = tid; localIdx < TILE_DIM; localIdx += threadsPerGroup) {
                    uint i_ = iStart + localIdx;
                    if (i_ < inputDim) {
                        atomicAdd(&(inputErrors[inOff_j + i_]), partial1D[localIdx]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // gradWeightsV => outer(inVal, dV_s[j_]) in tiles
            for (uint dStart = 0; dStart < headDim; dStart += TILE_DIM) {
                for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
                    for (uint idx = tid; idx < TILE_DIM*TILE_DIM; idx += threadsPerGroup) {
                        partialTile[idx] = 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                        uint ld = localIdx / TILE_DIM;
                        uint li = localIdx % TILE_DIM;
                        uint d_ = dStart + ld;
                        uint i_ = iStart + li;
                        if (d_ < headDim && i_ < inputDim) {
                            float val = inVal[i_] * dV_s[j_*headDim + d_];
                            partialTile[localIdx] = val;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                        uint ld = localIdx / TILE_DIM;
                        uint li = localIdx % TILE_DIM;
                        uint d_ = dStart + ld;
                        uint i_ = iStart + li;
                        if (d_ < headDim && i_ < inputDim) {
                            uint col = (dOff + d_);
                            atomicAdd(&(gradWeightsV[i_*modelDim + col]), partialTile[localIdx]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        } // end for j_ in [seqLength]

    } // end for h in [0..numHeads)

    // Done.
}



