#define REDUCTION_SUM     0
#define REDUCTION_MEAN    1
#define REDUCTION_MAX     2
#define REDUCTION_MIN     3
#define REDUCTION_SOFTMAX 4


#define ACTIVATION_LINEAR  0
#define ACTIVATION_RELU    1
#define ACTIVATION_TANH    2
#define ACTIVATION_SIGMOID 3
#define ACTIVATION_SOFTMAX 4
#define ACTIVATION_GELU    5


#ifndef common_metal
#define common_metal

#include <metal_stdlib>



inline float gelu(float x) {
    const float kAlpha = 0.7978845608f;  // sqrt(2/pi)
    
    float gelu_result = 0.5f * x * (1.0f + metal::tanh(kAlpha * (x + 0.044715f * metal::pow(x, 3))));
    
    // Debugging explicitly (temporarily)
    if (metal::abs(gelu_result) > 1e6) {
        gelu_result = metal::clamp(gelu_result, -10.0f, 10.0f);  // explicitly clamp to catch extreme values
    }
    
    return gelu_result;
}

inline float gelu_derivative(float x) {
    const float kAlpha = 0.7978845608f; // sqrt(2/pi)
    float x_cube = x * x * x;
    float tanh_arg = kAlpha * (x + 0.044715f * x_cube);
    float tanh_out = metal::tanh(tanh_arg);

    float left = 0.5f * (1.0f + tanh_out);
    float sech_sq = 1.0f - tanh_out * tanh_out;
    float right = 0.5f * x * sech_sq * kAlpha * (1.0f + 0.134145f * x * x);

    return left + right;
}


/**
 * A utility function to apply an activation.
 *
 * x: the input activation value
 * act: which activation to apply (e.g. ACTIVATION_RELU)
 */
inline float activate(const float x, const uint act) {
    switch (act) {
        case ACTIVATION_LINEAR:  return x;
        case ACTIVATION_RELU:    return metal::max(0.0f, x);
        case ACTIVATION_TANH:    return metal::tanh(x);
        case ACTIVATION_SIGMOID: return 1.0f / (1.0f + metal::exp(-x));
        case ACTIVATION_GELU:    return gelu(x);
        default:                 return 0.0f;  // Fallback
    }
}

/**
 * Derivative of the activation function.
 *
 * y: the already-activated value (e.g. y = activate(x, act))
 * act: which activation (e.g. ACTIVATION_RELU)
 */
inline float activate_derivative(const float y, const uint act) {
    switch (act) {
        case ACTIVATION_LINEAR:  return 1.0f;
        case ACTIVATION_RELU:    return (y > 0.0f) ? 1.0f : 0.0f;
        case ACTIVATION_TANH:    return 1.0f - (y * y);
        case ACTIVATION_SIGMOID: return y * (1.0f - y);
        case ACTIVATION_GELU:    return gelu_derivative(y);
        default:                 return 0.0f;
    }
}

#endif
