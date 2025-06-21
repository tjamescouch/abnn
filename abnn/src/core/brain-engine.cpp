// brain-engine.cpp – reward-modulated ABNN with teacher forcing
// ======================================================================

#include "brain-engine.h"
#include "brain.h"                 //  <-- needed for SynapsePacked & members
#include <random>                  //  <-- std::mt19937, distributions
#include <filesystem>
#include <fstream>
#include <mach-o/dyld.h>
#include <limits.h>
#include <numeric>
#include <chrono>
#include <thread>
#include <iostream>
#include "logger.h"
#include "stimulus-provider.h"
#include "common.h"


namespace fs = std::filesystem;

int step = 0;

/* helpers for paths ---------------------------------------------------- */
static fs::path data_path(const std::string& f){ return fs::current_path()/f; }
static fs::path bundle_resource(const std::string& f){
    char exe[PATH_MAX]; uint32_t n=sizeof(exe); _NSGetExecutablePath(exe,&n);
    return fs::canonical(exe).parent_path().parent_path()/ "Resources"/f;}

inline int clamp(float x, float min, float max) {
    if (x > max) return max;
    if (x < min) return min;
    return x;
}

inline double max(double x, double y) {
    return x > y ? x : y;
}

/* random graph --------------------------------------------------------- */
static void build_random_graph(Brain& b)
{
    static std::mt19937 gen(1);
    static std::uniform_real_distribution<float> wIn(0.8f,0.9f), wHH(0.1f,0.2f);

    auto* syn = reinterpret_cast<SynapsePacked*>(b.synapse_buffer()->contents());
    uint64_t idx=0,max=b.n_syn();

    /* dense input→output */
    for(uint32_t i=0;i<b.n_input() && idx<max;++i) {
        std::cout << ".";
        for(uint32_t o=0;o<b.n_output() && idx<max;++o) {
            syn[idx++] = { i, b.n_input()+o, wIn(gen), 0.f };
        }
    }
    std::cout << "X";

    /* sparse hidden */
    std::uniform_int_distribution<uint32_t> hid(
        b.n_input()+b.n_output(), b.n_neuron()-1);

    while(idx<max) {
        if (idx % 1000000 == 0)
            std::cout << 100 * idx / max << "%" << std::endl;
        syn[idx++] = { hid(gen), hid(gen), wHH(gen), 0.f };
    }

    b.synapse_buffer()->didModifyRange(NS::Range(0,max*sizeof(SynapsePacked)));
}

/* ctor / dtor ---------------------------------------------------------- */
BrainEngine::BrainEngine(MTL::Device* dev,
                         uint32_t nIn, uint32_t nOut,
                         uint32_t events)
: device_(dev->retain()),
  nIn_(nIn), nOut_(nOut), eventsPerPass_(events),
  spikeWindow_(nOut,0)
{
    commandQueue_ = device_->newCommandQueue();
    defaultLib_   = device_->newDefaultLibrary();

    brain_ = std::make_unique<Brain>(nIn_, nOut_, NUM_HIDDEN, NUM_SYN, eventsPerPass_);
    brain_->build_pipeline(device_, defaultLib_);
    brain_->build_buffers (device_);

    logger_ = std::make_unique<Logger>(nIn_, nOut_);

    if(!load_model()) {
        std::cout<<"🆕  building random graph…\n";
        build_random_graph(*brain_); save_model();
    }
}

BrainEngine::~BrainEngine(){
    stop_async();
    if(commandQueue_)commandQueue_->release();
    if(defaultLib_)  defaultLib_->release();
    if(device_)      device_->release(); }

/* model I/O ------------------------------------------------------------ */
bool BrainEngine::load_model(const std::string& nm){
    try {
        fs::path rw=data_path(nm.empty()?"model.bnn":nm);
        fs::path ro=bundle_resource("model.bnn");
        fs::path f = fs::exists(rw)? rw: ro;
        if(!fs::exists(f)) return false;
        std::ifstream is(f,std::ios::binary);
        if(!is) return false;
        brain_->load(is); std::cout<<"✅ loaded \""<<f<<"\"\n"; return true;
    } catch (std::exception* e){
        return false;
    }
}

bool BrainEngine::save_model(const std::string& nm) const{
    fs::path p=data_path(nm.empty()?"model.bnn":nm);
    std::ofstream os(p,std::ios::binary); if(!os) return false;
    brain_->save(os); std::cout<<"💾 saved → \""<<p<<"\"\n"; return true;}

/* stimulus ------------------------------------------------------------- */
void BrainEngine::set_stimulus(std::shared_ptr<StimulusProvider> s){ stim_=std::move(s); }

/* single pass ---------------------------------------------------------- */
/* ===================================================================== */
/*  Drop-in replacement for BrainEngine::run_one_pass()                  */
/*  - Fixes                                                               */
/*    • clamps tRate at 0.05 (not 0.30)                                   */
/*    • stronger reward gain + wider clip                                 */
/*    • correct best-loss tracking (min, not max)                         */
/*    • teacher-rate decays once every 10 windows                         */
/*    • shared-buffer writes marked with didModifyRange                   */
/* ===================================================================== */
std::vector<bool> BrainEngine::run_one_pass()
{
    if (!stim_) return {};

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    /* ---------- input & teacher spikes ------------------------------ */
    auto in       = stim_->nextInput();
    auto expected = stim_->nextExpected();
    brain_->inject_inputs(in, INPUT_RATE_HZ);

    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    uint32_t* lf  = static_cast<uint32_t*>(brain_->last_fired_buffer()->contents());
    uint32_t  now = *static_cast<uint32_t*>(brain_->clock_buffer()->contents());
    
    
    static float     exploreScale = 1.f;    // starts strong, will cool
    static uint64_t  rewardPasses = 0;        // counts reward-only passes

    static float     teacherRate = 1.0f;
    static uint64_t  block   = 0;
    static uint64_t  inBlock = 0;

    /* target value for this block */
    float target;
    switch (block) {
        case 0:  target = 1.0f; break;          // warm-up
        case 1:  target = 0.0f; break;
        case 2:  target = 1.0f; break;
        case 3:  target = 0.0f; break;
        case 4:  target = 0.5f; break;
        default: target = ((inBlock % 5000) < 1000) ? 0.5f : 0.0f; break;
    }

    /* never raise the current value — only lower it when target is lower */
    teacherRate = std::min(teacherRate, target);

    /* advance block counters (unchanged) */
    if (++inBlock == ((block <= 2) ? 1000 : (block == 3 ? 2000 : 2000))) {
        block++; inBlock = 0;
    }

    uint32_t passFlag = (teacherRate > 0.05f) ? PASS_TEACHER : PASS_REWARD;

    if (passFlag == PASS_REWARD)
        ++rewardPasses;

    const float kExploreFloor = 0.30f;    // new constant
    if (rewardPasses > 20000 && exploreScale > kExploreFloor)
            exploreScale = std::max(kExploreFloor, exploreScale * 0.99997f);


    /* ---------- teacher forcing ------------------------------------ */
    for (uint32_t o = 0; o < nOut_; ++o) {
        float p = expected[o] * teacherRate;
        if (uni(rng) < p && (now - lf[nIn_ + o] > 1))
            lf[nIn_ + o] = now;
    }

    /* ---------- GPU traversal -------------------------------------- */
    auto cb = commandQueue_->commandBuffer();
    
    
    
    brain_->setTeacherRate(teacherRate);
    
    brain_->setPassType(passFlag);          // buffer 15
    brain_->setExploreScale(exploreScale);  // buffer 16

    
    brain_->encode_traversal(cb);
    cb->commit();
    cb->waitUntilCompleted();


    /* ---------- outputs & debug snapshot --------------------------- */
    auto   spikes = brain_->read_outputs();
    DBG_OUT d     = brain_->read_debug_outputs();   // snapshot *then* counters reset
    
    //long nSpikes = std::count(spikes.begin(), spikes.end(), true);
    //std::cout << "Reward mode spikes: " << nSpikes << "\n";


    /* ---------- aggregate counters (1000-pass window) -------------- */
    static uint32_t winCtr      = 0;
    static double   hitsTeach   = 0.0 , dWteachAcc = 0.0;
    static double   hitsReward  = 0.0 , dWrewAcc   = 0.0;

    if (passFlag == PASS_TEACHER) {
        hitsTeach  += d.rewardHits;
        dWteachAcc += d.dwTeacher * 1e-6;
    } else {                        // PASS_REWARD
        hitsReward += d.rewardHits;
        dWrewAcc   += d.dwReward   * 1e-6;
    }

    if (++winCtr == 1000) {
        const bool haveTeach = (dWteachAcc > 0.0);
        const bool haveRew   = (dWrewAcc   > 0.0);

        if (haveTeach && haveRew) {
            double ratio = (dWteachAcc > 0.0) ? (dWrewAcc / dWteachAcc) : 0.0;

            std::cout << "♥ hitsT=" << hitsTeach
                      << " dWteach=" << dWteachAcc
                      << "  hitsR="  << hitsReward
                      << " dWrew="   << dWrewAcc
                      << "  ratio="  << ratio << '\n';

            if (ratio >= 0.15)
                teacherRate = std::max(teacherRate * 0.95f, 0.05f);

            hitsTeach = hitsReward = dWteachAcc = dWrewAcc = 0.0;
        }
        /* else: saw only one mode – keep accumulating */

        winCtr = 0;
    }

    /* ---------- smooth-rate logic (unchanged) ---------------------- */
    static std::vector<float> rate(nOut_, 0.f);
    const float alpha = 0.5f;
    for (uint32_t i = 0; i < nOut_; ++i)
        rate[i] = (1 - alpha) * rate[i] + alpha * (spikes[i] ? 1.f : 0.f);

    auto smoothRate = rateFilter_.process(rate, dT_SEC);

    for (auto r : smoothRate) maxObserved = std::max(maxObserved, r);
    maxObserved *= PEAK_DECAY;
    for (auto& r : smoothRate) r = std::min(r / maxObserved, 1.f);

    if ((++step % 100) == 0) logger_->log_samples(in, smoothRate);

    /* ---------- sliding-window loss & reward ----------------------- */
    for (uint32_t i = 0; i < nOut_; ++i) spikeWindow_[i] += spikes[i];
    ++winPos_;

    static double bestLoss = std::numeric_limits<double>::infinity();
    static double emaLoss  = 0.0;

    if (winPos_ == WIN_SIZE_) {
        double loss = 0.0;
        for (uint32_t i = 0; i < nOut_; ++i) {
            double e = smoothRate[i] - expected[i];
            loss += e * e;
        }
        loss /= nOut_;

        /* reward buffer -------------------------------------------- */
        float* rPtr = static_cast<float*>(brain_->reward_buffer()->contents());
        constexpr float kGain = 40.0f;
        float delta = static_cast<float>(lastLoss_ - loss);
        float tRate = std::max(teacherRate, 0.05f);
        float rVal  = std::clamp(kGain * delta / (tRate + 0.02f), -0.3f, 0.3f);
        *rPtr = rVal;
        brain_->reward_buffer()->didModifyRange(NS::Range(0, sizeof(float)));

        lastLoss_ = loss;
        logger_->accumulate_loss(loss);

        bestLoss = std::min(bestLoss, loss);
        emaLoss  = 0.98 * emaLoss + 0.02 * loss;

        std::cout << "🧑‍🏫 Teacher rate: " << teacherRate << '\n';
        winPos_ = 0;
    }
    
    static int rewardStep = 0;
    static int rewardSpikesTotal = 0;

    if (passFlag == PASS_REWARD) {
        rewardSpikesTotal += std::count(spikes.begin(), spikes.end(), true);
        if (++rewardStep == 1000) {
            std::cout << "📉 Avg reward spikes (last 1000): "
                      << rewardSpikesTotal / 1000.0 << "\n";
            rewardStep = 0;
            rewardSpikesTotal = 0;
        }
    }

    pool->release();
    return spikes;
}



/* async loop ----------------------------------------------------------- */
void BrainEngine::start_async(){
    if(running_.load()||!stim_) return;
    running_.store(true);
    worker_=std::thread([this]{
        while(running_.load()){
            run_one_pass();
        }
    });
    std::cout<<"▶️ Engine async loop started\n";
}

void BrainEngine::stop_async(){
    if(!running_.load()) return;
    running_.store(false);
    if(worker_.joinable()) worker_.join();
    std::cout<<"⏹️ Engine async loop stopped\n";
}
