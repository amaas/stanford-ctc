#include "decoder.h"
#include <fstream>
#include <iostream>
#include <boost/timer/timer.hpp>

// Mainly for profiling purposes

int main(int argc, char** argv)
{
    int N = 32;
    int T = 648;
    double data[N*T];

    std::ifstream is("probs.dat", std::ios::binary);
    is.read(reinterpret_cast<char*>(data), is.tellg());
    is.close();

    // TODO FIXME PARAM Import
    std::string CHARMAP_PATH = "/scail/group/deeplearning/speech/zxie/kaldi-stanford/kaldi-trunk/egs/wsj/s6/ctc-utils";
    PrefixTree ptree(CHARMAP_PATH + "/chars.txt",
                     CHARMAP_PATH + "/wordlist",
                     CHARMAP_PATH + "/lm_bg.arpa");

    // Decode!
    PrefixKey best_prefix;
    int beam = 40;
    double alpha = 1.0;
    double beta = 0.0;
    boost::timer::cpu_timer timer;
    double best_score = decode_bg_lm(data, N, T, ptree, beam, alpha, beta, best_prefix);
    boost::timer::cpu_times elapsed = timer.elapsed();
    std::cout << "decoding time (wall): " << elapsed.wall / 1e9 << std::endl;

    for (int k = 0; k < best_prefix.size(); k++)
    {
        std::cout << best_prefix[k] << " ";
    }
    std::cout << std::endl;

    return 0;
}
