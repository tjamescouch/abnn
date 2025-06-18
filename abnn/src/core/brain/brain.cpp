// brain.cpp  –  Host implementation of event-driven ABNN
// ======================================================================

#include "brain.h"

#include <cassert>
#include <cstring>
#include <random>
#include <cmath>

#include "constants.h"

/* helper: release Metal obj */
template<typename T> static void rel(T*& p){ if(p){ p->release(); p=nullptr; } }

std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<float> uni(0.f,1.f);

/* ===================================================================== */
/* ctor / dtor                                                           */
Brain::Brain(uint32_t nIn,uint32_t nOut,uint32_t nHid,
             uint32_t nSyn,uint32_t events)
: N_INPUT_(nIn), N_OUTPUT_(nOut), N_HIDDEN_(nHid),
  N_NRN_(nIn+nOut+nHid), N_SYN_(nSyn), EVENTS_(events),
  hostSyn_(nSyn)
{}
Brain::~Brain(){ release_all(); }

void Brain::release_all()
{
    rel(bufSyn_); rel(bufLastFire_); rel(bufLastVisit_);
    rel(bufClock_); rel(bufBudget_); rel(bufReward_); rel(bufRBar_);
    rel(pipeTrav_); rel(pipeRenorm_);
}

/* ===================================================================== */
/* build Metal pipeline objects                                          */
void Brain::build_pipeline(MTL::Device* d, MTL::Library* lib)
{
    NS::Error* e=nullptr;
    auto fnTrav = lib->newFunction(
        NS::String::string("monte_carlo_traversal",NS::UTF8StringEncoding));
    pipeTrav_ = d->newComputePipelineState(fnTrav,&e); fnTrav->release();

    auto fnRen = lib->newFunction(
        NS::String::string("renormalise_clock_and_times",NS::UTF8StringEncoding));
    pipeRenorm_= d->newComputePipelineState(fnRen,&e); fnRen->release();
}

/* ===================================================================== */
/* allocate & zero buffers                                               */
void Brain::build_buffers(MTL::Device* d)
{
    bufSyn_       = d->newBuffer(N_SYN_*sizeof(SynapsePacked), MTL::ResourceStorageModeManaged);
    bufLastFire_  = d->newBuffer(N_NRN_*sizeof(uint32_t),      MTL::ResourceStorageModeShared);
    bufLastVisit_ = d->newBuffer(N_NRN_*sizeof(uint32_t),      MTL::ResourceStorageModeShared);
    bufClock_     = d->newBuffer(sizeof(uint32_t),             MTL::ResourceStorageModeShared);
    bufBudget_    = d->newBuffer(sizeof(uint32_t),             MTL::ResourceStorageModeManaged);
    bufReward_    = d->newBuffer(sizeof(float),                MTL::ResourceStorageModeManaged);
    bufRBar_      = d->newBuffer(sizeof(float),                MTL::ResourceStorageModeShared);

    std::memset(bufSyn_->contents(),       0, bufSyn_->length());
    std::memset(bufLastFire_->contents(),  0, bufLastFire_->length());
    std::memset(bufLastVisit_->contents(), 0, bufLastVisit_->length());
    *static_cast<uint32_t*>(bufClock_->contents())  = 0;
    *static_cast<uint32_t*>(bufBudget_->contents()) = kMaxSpikes;
    *static_cast<float*>   (bufReward_->contents()) = 0.0f;
    *static_cast<float*>   (bufRBar_->contents())   = 0.0f;
}

/* ===================================================================== */
/* inject Poisson input spikes                                           */
void Brain::inject_inputs(const std::vector<float>& v, float hz)
{
    assert(v.size()==N_INPUT_);
    float pTick = hz * kTickNS * NSEC_PER_SEC;

    uint32_t* lf  = static_cast<uint32_t*>(bufLastFire_->contents());
    uint32_t  now = *static_cast<uint32_t*>(bufClock_->contents());

    for(uint32_t i=0;i<N_INPUT_;++i)
        if(uni(rng) < pTick * v[i]) lf[i] = now;
}

/* ===================================================================== */
/* encode one traversal                                                  */
void Brain::encode_traversal(MTL::CommandBuffer* cb)
{
    /* reset global spike budget */
    *static_cast<uint32_t*>(bufBudget_->contents()) = kMaxSpikes;
    bufBudget_->didModifyRange(NS::Range(0,sizeof(uint32_t)));

    auto enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(pipeTrav_);

    enc->setBuffer(bufSyn_,       0, 0);
    enc->setBuffer(bufLastFire_,  0, 1);
    enc->setBuffer(bufLastVisit_, 0, 2);
    enc->setBuffer(bufClock_,     0, 3);
    enc->setBytes (&N_SYN_, sizeof(uint32_t), 4);

    uint32_t tauVis=50'000, tauPre=50'000;
    enc->setBytes(&tauVis,sizeof(uint32_t),5);
    enc->setBytes(&tauPre,sizeof(uint32_t),6);

    float aLTP=_aLTP, aLTD=_aLTD, wMin= _wMin, wMax=_wMax;
    enc->setBytes(&aLTP,sizeof(float),7);
    enc->setBytes(&aLTD,sizeof(float),8);
    enc->setBytes(&wMin,sizeof(float),9);
    enc->setBytes(&wMax,sizeof(float),10);

    enc->setBuffer(bufBudget_, 0, 11);
    enc->setBuffer(bufReward_, 0, 12);
    enc->setBuffer(bufRBar_,   0, 13);

    const uint tg=256;
    enc->dispatchThreads(MTL::Size(((EVENTS_+tg-1)/tg)*tg,1,1),
                         MTL::Size(tg,1,1));
    enc->endEncoding();

    renormalise_if_needed(cb);
}

/* ===================================================================== */
void Brain::renormalise_if_needed(MTL::CommandBuffer* cb)
{
    uint32_t now=*static_cast<uint32_t*>(bufClock_->contents());
    if(now<=kRenormThresh) return;

    auto enc=cb->computeCommandEncoder();
    enc->setComputePipelineState(pipeRenorm_);
    enc->setBuffer(bufLastFire_,0,0);
    enc->setBuffer(bufLastVisit_,0,1);
    enc->setBuffer(bufClock_,0,2);
    enc->setBytes(&N_NRN_,sizeof(uint32_t),3);

    const uint tg=256;
    enc->dispatchThreads(MTL::Size(((N_NRN_+tg-1)/tg)*tg,1,1),
                         MTL::Size(tg,1,1));
    enc->endEncoding();
}

/* ===================================================================== */
/* read output spikes (bool vector)                                      */
std::vector<bool> Brain::read_outputs() const
{
    std::vector<bool> out(N_OUTPUT_,false);
    const uint32_t* lf = static_cast<const uint32_t*>(bufLastFire_->contents());
    uint32_t now = *static_cast<const uint32_t*>(bufClock_->contents());

    uint32_t start=now>1? now-1:0;
    for(uint32_t o=0;o<N_OUTPUT_;++o){
        uint32_t ts=lf[N_INPUT_+o];
        if(ts!=0 && ts>=start && ts<now) out[o]=true;
    }
    return out;
}

/* ===================================================================== */
/* persistence                                                           */
void Brain::save(std::ostream& os) const
{
    os.write(reinterpret_cast<const char*>(&N_SYN_), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&N_NRN_), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(bufSyn_->contents()),
             N_SYN_*sizeof(SynapsePacked));
}

void Brain::load(std::istream& is)
{
    uint32_t s{},n{};
    is.read(reinterpret_cast<char*>(&s),4);
    is.read(reinterpret_cast<char*>(&n),4);
    if (!(s==N_SYN_ && n==N_NRN_)) throw new std::exception();
    is.read(reinterpret_cast<char*>(bufSyn_->contents()),
            N_SYN_*sizeof(SynapsePacked));
    bufSyn_->didModifyRange(NS::Range(0,N_SYN_*sizeof(SynapsePacked)));
}
