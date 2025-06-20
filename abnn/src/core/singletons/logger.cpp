#include "logger.h"
#include <filesystem>
#include <iostream>
#include <numeric>

namespace fs = std::filesystem;
const std::string filename = "abnn_session.m";

/* ctor: open session file, write header */
Logger::Logger(int nIn, int nOut)
: nIn_(nIn), nOut_(nOut)
{
    fs::path p = fs::current_path() / filename;
    mat_.open(p, std::ios::trunc);
    if (!mat_)
        std::cerr << "❌ cannot open " << p << '\n';
    else
        mat_ << "% ABNN animated session\n";
}

/* dtor: close file */
Logger::~Logger() { if(mat_) mat_.close(); }

/* animate one frame ------------------------------------------------------ */
void Logger::log_samples(const std::vector<float>& in,
                         const std::vector<float>& out)
{
    if(!mat_) return;
    
    // input
    mat_ << "clf;\nhold on;\nylim([-1 1]);\n";
    
    mat_ << "xo = [ ";
    for(size_t i=0;i<nOut_;++i){ mat_<<i; mat_<<" "; }
    mat_ << "];\n";
    
    mat_ << "x = [ ";
    for(size_t i=0;i<in.size();++i){ mat_<<i; mat_<<" "; }
    mat_ << "];\n";
    
    mat_ << "y = [ ";
    for(size_t i=0;i<in.size();++i){ mat_<<in[i]; mat_<<" "; }
    mat_ << "];\n";


    // output
    mat_ << "\nz=[";
    for(size_t i=0;i<out.size();++i){ if(i) mat_<<","; mat_<<out[i]; }
    mat_ << "];title('Output');\n";

    mat_ << "scatter(x,y,[],[],[0,0,1]);\n";
    mat_ << "scatter(xo,z,[],[],[0,1,0]);\n";
    mat_ << "hold off; pause(0.03);\n\n";
    
    mat_.flush();
}

/* loss EMA --------------------------------------------------------------- */
void Logger::accumulate_loss(double loss)
{
    if(step_==0) ema_ = loss;
    else         ema_ = beta_*ema_ + (1.0-beta_)*loss;
    ++step_;

    std::setprecision(5);
    std::cout << "✨ EMA-Loss: " << ema_ << " ❌ Raw loss: " << loss << std::endl;
    
    if(step_ % 10 == 0) flush();
}

void Logger::flush()
{
    std::cout << "📈 Truncating output graph file" << '\n';
    
    mat_.flush();
    mat_.close();
    
    fs::path p = fs::current_path() / filename;
    mat_.open(p, std::ios::trunc);
    if (!mat_)
        std::cerr << "❌ cannot open " << p << '\n';
    else
        mat_ << "% ABNN animated session\n";
}
