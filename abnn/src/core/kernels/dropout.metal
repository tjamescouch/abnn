#include <metal_stdlib>
using namespace metal;

#include "common.metal"


float activate(const float x, const uint act);
float activate_derivative(const float y, const uint act);

 kernel void forward_dropout(
     device const float* input       [[buffer(0)]],
     device       float* output      [[buffer(1)]],
     device const float* randomMask  [[buffer(2)]], // binary mask: 0 or 1
     constant float&     rate        [[buffer(3)]],
     constant int&       featureDim  [[buffer(4)]],
     constant bool&      isTraining  [[buffer(5)]],
     device float*       debug       [[buffer(6)]],
     uint                tid         [[thread_position_in_grid]]
 ) {
     if ((int)tid >= featureDim) return;
 
     float x        = input[tid];
     float maskVal  = randomMask[tid]; // binary mask value

     if (isTraining) {
         output[tid] = (maskVal > 0.5f) ? (x / (1.0f - rate)) : 0.0f;
     } else {
         output[tid] = x;
     }
     debug[tid] = input[tid];
 }

//-------------------------------------------------------------------
// Backward pass for Dropout layer
 kernel void backward_dropout(
     device const float* input_error  [[buffer(0)]],
     device float*       output_error [[buffer(1)]],
     device const float* randomMask   [[buffer(2)]],
     constant float&     rate         [[buffer(3)]],
     constant uint&      featureDim   [[buffer(4)]],
     device const float*       debug        [[buffer(5)]],
     uint tid                         [[thread_position_in_grid]]
 ) {
     if (tid >= featureDim) return;
 
     float maskVal  = randomMask[tid]; // binary mask value
     output_error[tid] = (maskVal > 0.5) ? (input_error[tid] / (1.0f - rate)) : 0.0f;
 }
