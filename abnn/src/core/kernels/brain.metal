#include <metal_stdlib>
using namespace metal;

struct SynapsePacked { uint src,dst; float w,pad; };
inline float rand01(uint s){ s^=s<<13; s^=s>>17; s^=s<<5;
                              return (float)(s&0xFFFFFF)/16777216.0f; }

#define BASE_SCALE   0.2f
#define REFRACTORY   20u
#define WINDOW_PRE    5u     /* ← 5-tick presyn window  */
#define MAX_SPIKES   16u
#define CLOCK_INC     1u

kernel void monte_carlo_traversal(
    device const SynapsePacked* syn[[buffer(0)]],
    device atomic_uint* lastF[[buffer(1)]],
    device atomic_uint* lastV[[buffer(2)]],
    device atomic_uint* clk  [[buffer(3)]],
    constant uint& nSyn      [[buffer(4)]],
    constant uint& tau_v     [[buffer(5)]],
    constant uint& tau_p     [[buffer(6)]],
    constant float& aLTP     [[buffer(7)]],
    constant float& aLTD     [[buffer(8)]],
    constant float& wMin     [[buffer(9)]],
    constant float& wMax     [[buffer(10)]],
    device atomic_uint* budg [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    if(tid>=nSyn) return;
    SynapsePacked s=syn[tid];
    uint now=atomic_load_explicit(clk,memory_order_relaxed);

    /* presyn gate */
    uint lp=atomic_load_explicit(&lastF[s.src],memory_order_relaxed);
    if(now-lp>WINDOW_PRE){ if(tid==0) atomic_fetch_add_explicit(clk,CLOCK_INC,memory_order_relaxed); return; }

    /* refractory */
    if(now-atomic_load_explicit(&lastF[s.dst],memory_order_relaxed)<=REFRACTORY){
        if(tid==0) atomic_fetch_add_explicit(clk,CLOCK_INC,memory_order_relaxed); return;
    }

    /* budget check */
    if(atomic_load_explicit(budg,memory_order_relaxed)==0u){
        if(tid==0) atomic_fetch_add_explicit(clk,CLOCK_INC,memory_order_relaxed); return;
    }

    float p=clamp(s.w*s.w*BASE_SCALE,0.f,1.f);
    bool fired=(p>rand01(tid^now));
    if(fired){
        uint old=atomic_fetch_sub_explicit(budg,1u,memory_order_relaxed);
        if(old==0u) fired=false;
    }

    float dW=fired? aLTP*(1.f-s.w) : -aLTD*s.w;
    s.w=clamp(s.w+dW,wMin,wMax);
    ((device SynapsePacked*)syn)[tid].w=s.w;

    if(fired) atomic_store_explicit(&lastF[s.dst],now,memory_order_relaxed);
    if(tid==0) atomic_fetch_add_explicit(clk,CLOCK_INC,memory_order_relaxed);
}

kernel void renormalise_clock_and_times(
    device atomic_uint* lf[[buffer(0)]],
    device atomic_uint* lv[[buffer(1)]],
    device atomic_uint* clk[[buffer(2)]],
    constant uint& n  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint base=atomic_load_explicit(clk,memory_order_relaxed);
    if(tid<n) atomic_fetch_sub_explicit(&lf[tid],base,memory_order_relaxed);
    if(tid==0) atomic_store_explicit(clk,0u,memory_order_relaxed);
}
