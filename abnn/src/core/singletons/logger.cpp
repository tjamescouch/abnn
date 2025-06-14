#include "logger.h"
#include <filesystem>
#include <iostream>
#include <numeric>

namespace fs = std::filesystem;

/* ctor: open session file, write header */
Logger::Logger(int nIn, int nOut)
: nIn_(nIn), nOut_(nOut)
{
    fs::path p = fs::current_path() / "abnn_session.m";
    mat_.open(p, std::ios::trunc);
    if (!mat_)
        std::cerr << "❌ cannot open " << p << '\n';
    else
        mat_ << "% ABNN animated session\n";
    
    mat_ << "x = linspace(0, " << nIn - 1 << ");\n";
}

/* dtor: close file */
Logger::~Logger() { if(mat_) mat_.close(); }

/* animate one frame ------------------------------------------------------ */
void Logger::log_samples(const std::vector<float>& in,
                         const std::vector<float>& out)
{
    if(!mat_) return;
    
    

    /* input */
    mat_ << "clf;\nsubplot(2,1,1);\ny = [ ";
    for(size_t i=0;i<in.size();++i){ if(i) mat_<<","; mat_<<in[i]; }
    mat_ << " ];\nplot(x,y);\nylim([0 1]);\ntitle('Input');\n";

    /* output
    mat_ << "subplot(2,1,2);\nplot(";
    for(size_t i=0;i<out.size();++i){ if(i) mat_<<","; mat_<<out[i]; }
    mat_ << ", 'r-'); ylim([0 1]); title('Output');\n";
    mat_ << "drawnow; pause(0.03);\n\n";
    mat_.flush();*/
}

/* loss EMA --------------------------------------------------------------- */
void Logger::accumulate_loss(double loss)
{
    if(step_==0) ema_ = loss;
    else         ema_ = beta_*ema_ + (1.0-beta_)*loss;
    ++step_;

    if(step_ % 200 == 0) flush_loss();
}

void Logger::flush_loss()
{
    std::cout << "✨ EMA-Loss: " << ema_ << '\n';
}
