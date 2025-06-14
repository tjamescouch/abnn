// brain-engine.cpp – asynchronous harness driving Brain
// =====================================================

#include "brain-engine.h"
#include "logger.h"
#include "functional-dataset.h"      // your sine stimulus

#include <filesystem>
#include <fstream>
#include <mach-o/dyld.h>
#include <limits.h>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

/* bundle resource → path -------------------------------------------------- */
static fs::path bundle_resource(const std::string& file)
{
    char exe[PATH_MAX]; uint32_t n=sizeof(exe);
    _NSGetExecutablePath(exe,&n);
    return fs::canonical(exe).parent_path().parent_path()/"Resources"/file;
}
/* sandbox data path ------------------------------------------------------- */
static fs::path data_path(const std::string& file){ return fs::current_path()/file; }

/* random graph builder (safe) -------------------------------------------- */
static void build_random_graph(Brain& b)
{
    std::mt19937 gen(1);
    std::uniform_real_distribution<float> wIn(0.5f,0.8f), wHH(0.1f,0.2f);
    auto* syn = reinterpret_cast<SynapsePacked*>(b.synapse_buffer()->contents());
    uint32_t idx=0,max=b.n_syn();

    for(uint32_t i=0;i<b.n_input() && idx<max;++i)
        for(uint32_t o=0;o<b.n_output() && idx<max;++o)
            syn[idx++]={i,b.n_input()+o,wIn(gen),0.f};

    std::uniform_int_distribution<uint32_t> hid(b.n_input()+b.n_output(), b.n_neuron()-1);
    while(idx<max) syn[idx++]={hid(gen),hid(gen),wHH(gen),0.f};

    b.synapse_buffer()->didModifyRange(NS::Range(0,max*sizeof(SynapsePacked)));
}

/* ctor / dtor ------------------------------------------------------------- */
BrainEngine::BrainEngine(MTL::Device* dev,
                         uint32_t nIn, uint32_t nOut, uint32_t events)
: device_(dev->retain()), nIn_(nIn), nOut_(nOut), eventsPerPass_(events)
{
    commandQueue_=device_->newCommandQueue();
    defaultLib_  =device_->newDefaultLibrary();

    brain_=std::make_unique<Brain>(nIn_,nOut_,128,4096,eventsPerPass_);
    brain_->build_pipeline(device_,defaultLib_);
    brain_->build_buffers (device_);

    logger_=std::make_unique<Logger>(nIn_,nOut_);

    if(!load_model()){
        std::cout<<"🆕 building random graph…\n";
        build_random_graph(*brain_); save_model();
    }
}
BrainEngine::~BrainEngine(){ stop_async();
    if(commandQueue_)commandQueue_->release();
    if(defaultLib_)  defaultLib_->release();
    if(device_)      device_->release(); }

/* model I/O --------------------------------------------------------------- */
bool BrainEngine::load_model(const std::string& nm)
{
    fs::path rw=data_path(nm.empty()?"model.bnn":nm);
    fs::path ro=bundle_resource("model.bnn");
    fs::path f = fs::exists(rw)? rw : ro;
    if(!fs::exists(f)) return false;
    std::ifstream is(f,std::ios::binary); if(!is) return false;
    brain_->load(is); std::cout<<"✅ loaded \""<<f<<"\"\n"; return true;
}
bool BrainEngine::save_model(const std::string& nm)const
{
    fs::path p=data_path(nm.empty()?"model.bnn":nm);
    std::ofstream os(p,std::ios::binary); if(!os) return false;
    brain_->save(os); std::cout<<"💾 saved → \""<<p<<"\"\n"; return true;
}

/* stimulus ---------------------------------------------------------------- */
void BrainEngine::set_stimulus(std::shared_ptr<StimulusProvider> s){ stim_=std::move(s); }

/* synchronous pass (incl. loss) ------------------------------------------ */
std::vector<bool> BrainEngine::run_one_pass()
{
    if(!stim_) return {};
    auto in = stim_->next();              /* sine frame */
    brain_->inject_inputs(in, 1000.0f);   /* 1 kHz Poisson */

    auto cb=commandQueue_->commandBuffer();
    brain_->encode_traversal(cb); cb->commit(); cb->waitUntilCompleted();

    auto out = brain_->read_outputs();
    std::vector<float> outF(out.begin(), out.end());
    logger_->log_samples(in,outF);

    /* loss = (mean input – spike fraction)² ------------------------------ */
    double muIn = std::accumulate(in.begin(),in.end(),0.0)/in.size();
    double nSpk = std::accumulate(out.begin(),out.end(),0);
    double muOut= nSpk / nOut_;
    double loss = (muIn - muOut)*(muIn - muOut);
    logger_->accumulate_loss(loss);

    return out;
}

/* async loop -------------------------------------------------------------- */
void BrainEngine::start_async()
{
    if(running_.load()||!stim_) return;
    running_.store(true);
    worker_=std::thread([this]{
        while(running_.load()){
            run_one_pass();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    std::cout<<"▶️ Engine async loop started\n";
}
void BrainEngine::stop_async()
{
    if(!running_.load()) return;
    running_.store(false);
    if(worker_.joinable()) worker_.join();
    std::cout<<"⏹️ Engine async loop stopped\n";
}
