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

/* random graph --------------------------------------------------------- */
static void build_random_graph(Brain& b)
{
    std::mt19937 gen(1);
    std::uniform_real_distribution<float> wIn(0.4f,0.8f),
                                          wHH(0.1f,0.2f);

    auto* syn = reinterpret_cast<SynapsePacked*>(b.synapse_buffer()->contents());
    uint32_t idx=0,max=b.n_syn();

    /* dense input→output */
    for(uint32_t i=0;i<b.n_input() && idx<max;++i)
        for(uint32_t o=0;o<b.n_output() && idx<max;++o)
            syn[idx++] = { i, b.n_input()+o, wIn(gen), 0.f };

    /* sparse hidden */
    std::uniform_int_distribution<uint32_t> hid(
        b.n_input()+b.n_output(), b.n_neuron()-1);

    while(idx<max)
        syn[idx++] = { hid(gen), hid(gen), wHH(gen), 0.f };

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

    if(!load_model()){ std::cout<<"🆕  building random graph…\n";
                       build_random_graph(*brain_); save_model(); }
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
std::vector<bool> BrainEngine::run_one_pass()
{
    if(!stim_) return {};
    
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    
    auto in = stim_->next();
    brain_->inject_inputs(in, INPUT_RATE_HZ);

    //–– Poisson teacher forcing: pTeach = in[o] * teacherRate
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<float> uni(0.0f,1.0f);

    uint32_t* lf  = (uint32_t*)brain_->last_fired_buffer()->contents();
    uint32_t  now = *(uint32_t*)brain_->clock_buffer()->contents();

    float teacherRate = 1.f;      // adjust so average teacher spikes ~20% of ticks
    for (uint32_t o = 0; o < nOut_; ++o) {
        float p = in[o] * teacherRate;
        if (uni(rng) < p && (now - lf[nIn_ + o] > 1)) {
            lf[nIn_ + o] = now;    // inject a “teacher” spike
        }
    }

    auto cb=commandQueue_->commandBuffer();

    brain_->encode_traversal(cb);
    
    cb->commit();
    cb->waitUntilCompleted();

    auto out = brain_->read_outputs();

    static std::vector<float> rate(nOut_, 0.f);
    const float alpha = 0.5f;                // smoothing factor

    auto spikes = brain_->read_outputs();
    for (int i = 0; i < nOut_; ++i) {
        rate[i] = (1-alpha)*rate[i] + alpha*(spikes[i]?1.f:0.f);
    }

    auto smoothRate = rateFilter_.process(rate, dT_SEC);
    
    // after computing smoothRate:
    for (auto r : smoothRate) {
        maxObserved = std::max(maxObserved, r);
    }
    maxObserved *= PEAK_DECAY;               // slowly forget old peaks

    // now normalize:
    for (auto &r : smoothRate) {
        r = std::min(r / maxObserved, 1.0f);
    }

    if ((++step % 100) == 0) {
        logger_->log_samples(in, smoothRate);
    }
    
    /* sliding window */
    for(uint32_t i=0;i<nOut_;++i) spikeWindow_[i]+= out[i]?1:0;
    ++winPos_;
    if(winPos_==WIN_SIZE_) {
        double loss = 0.0;
        for (uint32_t i = 0; i < nOut_; ++i) {
            double err = smoothRate[i] - in[i];
            loss += err * err;
        }
        loss /= nOut_;
        float* r=(float*)brain_->reward_buffer()->contents();
        *r = float(lastLoss_ - loss);
        brain_->reward_buffer()->didModifyRange(NS::Range(0,4));
        lastLoss_=loss;
        logger_->accumulate_loss(loss);
        winPos_=0;
    }
    
    pool->release();
    return out;
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
