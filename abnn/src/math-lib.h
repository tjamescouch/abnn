//
//  math-lib.h
//  MetalNN
//
//  Created by James Couch on 2025-02-16.
//

#ifndef MATH_LIB_H
#define MATH_LIB_H

#include <simd/simd.h>

namespace mathlib {

const inline float kPi = 3.14159265358979323846;

inline float radians(float degrees) {
    return degrees * (kPi / 180.0);
}

inline simd::float4x4 makeProjectionMatrix(float fov, float aspect, float near, float far) {
    float yScale = 1.0 / tan(fov * 0.5);
    float xScale = yScale / aspect;
    float zScale = -(far + near) / (far - near);
    float zTranslation = -(2.0 * far * near) / (far - near);
    
    return simd::float4x4(
                          simd::float4{xScale, 0.0,    0.0,     0.0},
                          simd::float4{0.0,    yScale, 0.0,     0.0},
                          simd::float4{0.0,    0.0,    zScale,  -1.0},
                          simd::float4{0.0,    0.0,    zTranslation, 0.0}
                          );
}

//------------------------------------------------------------------------------
// 1) lookAtMatrix(eye, center, up)
//
// Produces a view matrix that transforms world-space points into camera space,
// looking from 'eye' toward 'center' with 'up' indicating the world-space up axis.
//
// By default, this uses a right-handed convention, so the camera looks down
// the -Z axis in camera space.
//------------------------------------------------------------------------------
inline simd::float4x4 lookAtMatrix(simd::float3 eye,
                                   simd::float3 center,
                                   simd::float3 up)
{
    // Forward direction (camera looks from eye toward center)
    simd::float3 f = simd::normalize(center - eye);
    // Right vector
    simd::float3 s = simd::normalize(simd::cross(f, up));
    // Actual up vector
    simd::float3 u = simd::cross(s, f);
    
    // Construct a column-major matrix
    // (The negative sign on 'f' in the third column makes +Z point *into* the scene)
    
    return simd::float4x4{
        simd::float4{ s.x,    u.x,   -f.x,   0},
        simd::float4{ s.y,    u.y,   -f.y,   0},
        simd::float4{ s.z,    u.z,   -f.z,   0},
        simd::float4{ -simd::dot(s, eye), -simd::dot(u, eye), simd::dot(f, eye), 1}
    };
}

//------------------------------------------------------------------------------
// 2) orthographicMatrix(left, right, bottom, top, nearZ, farZ)
//
// Produces a right-handed orthographic projection matrix, which maps
// the box [left..right] × [bottom..top] × [nearZ..farZ] into clip space.
// In clip space, X ∈ [-1..1], Y ∈ [-1..1], Z ∈ [-1..1].
//
// Adjust signs or coordinate system if you prefer a left-handed system
// or if your engine uses a different convention for Z.
//------------------------------------------------------------------------------
inline simd::float4x4 makeOrthographicMatrix(float left, float right, float bottom, float top, float nearZ, float farZ)
{
    float rl = right - left;
    float tb = top - bottom;
    float fn = farZ - nearZ;
    
    return simd::float4x4(
                          simd::float4{2.0f / rl, 0, 0, 0},
                          simd::float4{0, 2.0f / tb, 0, 0},
                          simd::float4{0, 0, -2.0f / fn, 0},
                          simd::float4{-(right + left) / rl, -(top + bottom) / tb, -(farZ + nearZ) / fn, 1.0f}
                          );
}


inline float fastExp2(float p)
{
    // clamp input range if needed
    if (p < -126.0f) p = -126.0f;
    if (p >  127.0f) p =  127.0f;
    
    // This union allows us to do bit-level manipulation
    // to approximate 2^p. The magic constant 12102203
    // is close to (1 << 23) * (1 / ln(2)).
    union { uint32_t i; float f; } v;
    v.i = (int)(12102203 * p + 127 * (1 << 23)) & 0x7FFFFF; // rough
    return v.f;
}

inline float fastExpf(float x)
{
    // e^x = 2^(x / log2(e)) => multiply x by 1.442695
    return fastExp2(1.442695f * x);
}



inline double inputFunc(double index, double timestep) {
    return sin(0.05 * index + 0.1 * timestep);
}

inline double targetFunc(double index, double timestep) {
    return cos(0.05 * index + 0.1 * timestep);
}

template <typename T> inline T min(T a, T b) {
    return a < b ? a : b;
}

template <typename T> inline T max(T a, T b) {
    return a > b ? a : b;
}

template <typename T> inline T clamp(T value, T min, T max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

}

#endif

