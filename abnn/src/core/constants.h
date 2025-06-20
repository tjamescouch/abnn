
#define NUM_INPUTS  256
#define NUM_OUTPUTS 256
#define NUM_HIDDEN 5'000'000
#define NUM_SYN    1'000'000'000

#define INPUT_SIN_WAVE_FREQUENCY 0.5

#define INPUT_RATE_HZ 1000
#define PEAK_DECAY 0.999f         // how quickly old peaks fade
#define EVENTS_PER_PASS 150'000'000
#define FILTER_TAU 0.02
#define USE_FIR true
#define dT_SEC 0.0009

#define _aLTP 0.04f
#define _aLTD 0.02f
#define _wMin 0.001f
#define _wMax 1.0f
