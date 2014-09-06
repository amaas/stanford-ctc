#include "decoder.h"
#include "lm.h"
#include <cmath>
#include <cassert>
#include <vector>
#include <map>
#include <fstream>
#include <boost/unordered_map.hpp>
#include <Python.h>
#include <boost/python/extract.hpp>
#include <numpy/noprefix.h>
#include <boost/timer/timer.hpp>

// NOTE
// Current bottlenecks according to gprof are
// std::sort
// unordered_map find
// unordered_map []


double combine(double a, double b, double c)
{
    double psum = exp(a) + exp(b) + exp(c);
    if (psum < EPS)
        return NEG_INF;
    return log(psum);
}


double prefix_hyp_score(PrefixHyp::Ptr hyp)
{
    return combine(hyp->p_nb, hyp->p_b) + COMP_BETA * hyp->nw;
}

// TODO Not why why trying to use PrefixPair& reference causes compiler to complain
bool comp_prefix_pair(const PrefixPair x, const PrefixPair y)
{
    // Want descending order
    //return prefix_hyp_score(x.second) > prefix_hyp_score(y.second);
    return x.second->score > y.second->score;
}

double decode_bg_lm(double* probs, int N, int T, PrefixTree& ptree,
        int beam, double alpha, double beta, PrefixKey& best_prefix)
{
    double bg = 0.0;
    float best_score = -1;

    PrefixMap Hold;
    PrefixMap Hcurr;
    PrefixMap Hnext;

    // Start things off
    std::vector<int> empty_prefix;
    PrefixHyp::Ptr empty_hyp(new PrefixHyp);
    empty_hyp->p_b = 0.0;
    empty_hyp->node = ptree.root;
    empty_hyp->wid = ptree.lm.start;
    empty_hyp->score = prefix_hyp_score(empty_hyp);
    Hcurr[empty_prefix] = empty_hyp;

    // Loop over time
    for (int t = 0; t < T; t++) {
        Hnext.clear();
        double probs_0t = probs[t*N+0];

        // Loop over prefixes
        for (PrefixMap::iterator iter=Hcurr.begin(); iter != Hcurr.end(); ++iter) {
            PrefixKey prefix = iter->first;
            PrefixHyp::Ptr hyp = iter->second;

            PrefixHyp::Ptr valsP(new PrefixHyp);
            if (Hnext.find(prefix) != Hnext.end())
                valsP = Hnext[prefix];
            else
                Hnext[prefix] = valsP;
            valsP->p_b = combine(hyp->p_nb + probs_0t, hyp->p_b + probs_0t, valsP->p_b);
            valsP->node = hyp->node;
            valsP->wid = hyp->wid;
            valsP->nw = hyp->nw;

            // Loop over characters
            // Here's where stuff needs to be optimized
            for (int i = 1; i < N; i++)
            {
                PrefixTreeNode::Ptr new_node;
                double probs_it = probs[t*N+i];

                bool skip = false;
                bool emit_word = false;
                if (i == ptree.space) {
                    if (hyp->node->is_word) {
                        emit_word = true;
                        new_node = ptree.root;
                    }
                    else {
                        skip = true;
                    }
                }
                else {
                    if (hyp->node->children.size() > 0 &&
                        hyp->node->children.find(i) != hyp->node->children.end() &&
                        (hyp->node->children[i]->is_prefix || hyp->node->children[i]->is_word)) {
                        new_node = hyp->node->children[i];
                    }
                    else
                        skip = true;
                }
                if (!skip) {
                    PrefixKey nprefix = prefix;
                    nprefix.push_back(i);
                    // TODO CHECK Assuming not yet in Hnext, may not be true
                    PrefixHyp::Ptr valsN(new PrefixHyp);
                    valsN->node = new_node;
                    if (Hnext.find(nprefix) != Hnext.end())
                        valsN = Hnext[nprefix];
                    else
                        Hnext[nprefix] = valsN;

                    if (emit_word) {
                        bg = alpha * ptree.lm.bg_prob(hyp->wid, hyp->node->word_id);
                        valsN->wid = hyp->node->word_id;
                        valsN->nw = hyp->nw + 1;
                    }
                    else {
                        bg = 0.0;
                        valsN->wid = hyp->wid;
                        valsN->nw = hyp->nw;
                    }

                    if (prefix.size() == 0 || (prefix.size() > 0 && i != prefix.back()))
                        valsN->p_nb = combine(hyp->p_nb + probs_it + bg, hyp->p_b + probs_it + bg, valsN->p_nb);
                    else
                        valsN->p_nb = combine(hyp->p_b + probs_it + bg, valsN->p_nb);

                    if (Hcurr.find(nprefix) == Hcurr.end()) {
                        PrefixHyp::Ptr oldhyp(new PrefixHyp);
                        if (Hold.find(nprefix) != Hold.end())
                            oldhyp = Hold[nprefix];
                        else
                            Hold[nprefix] = oldhyp;
                        valsN->p_b = combine(oldhyp->p_nb + probs_0t, oldhyp->p_b + probs_0t, valsN->p_b);
                        valsN->p_nb = combine(oldhyp->p_nb + probs_it, valsN->p_nb);
                    }
                    valsN->score = prefix_hyp_score(valsN);
                }
                if (prefix.size() > 0 && i == prefix.back()) {
                    valsP->p_nb = combine(hyp->p_nb + probs_it, valsP->p_nb);
                }

                valsP->score = prefix_hyp_score(valsP);
            }
        }

        Hold = Hnext;

        // Beam cutoff
        std::vector<PrefixPair> Hnext_vec(Hnext.begin(), Hnext.end());
        std::sort(Hnext_vec.begin(), Hnext_vec.end(), comp_prefix_pair);
        if (Hnext_vec.size() > beam)
            Hnext_vec.erase(Hnext_vec.begin() + beam, Hnext_vec.end());
        best_prefix = Hnext_vec[0].first;
        best_score = Hnext_vec[0].second->score;
        Hcurr = PrefixMap(Hnext_vec.begin(), Hnext_vec.end());
    }

    return best_score;
}

py::tuple decode_lm_wrapper(np::array probs, int beam, double alpha, double beta)
{
    // Check that we can use numeric_limits infinity, then comment out
    //assert(std::numeric_limits<double>::is_iec559);

    py::object shape = probs.attr("shape");
    unsigned int N = py::extract<unsigned int>(shape[0]);
    unsigned int T = py::extract<unsigned int>(shape[1]);
    std::cout << N << "x" << T << " probs" << std::endl;

    // Convert probs to double array, row-major
    double* data = static_cast<double*>(PyArray_DATA(probs.ptr()));

    // Write data for profiling in another script
    std::ofstream os("probs.dat", std::ios::binary);
    os.write(reinterpret_cast<const char*>(data), std::streamsize(N*T*sizeof(double)));
    os.close();

    // Build / load prefix tree
    // TODO Create a prefix tree server that keeps object in memory

    // TODO FIXME PARAM Import
    std::string CHARMAP_PATH = "/scail/group/deeplearning/speech/zxie/kaldi-stanford/kaldi-trunk/egs/wsj/s6/ctc-utils";
    PrefixTree ptree(CHARMAP_PATH + "/chars.txt",
                     CHARMAP_PATH + "/wordlist",
                     CHARMAP_PATH + "/lm_bg.arpa");

    // Decode!
    PrefixKey best_prefix;
    boost::timer::cpu_timer timer;
    double best_score = decode_bg_lm(data, N, T, ptree, beam, alpha, beta, best_prefix);
    boost::timer::cpu_times elapsed = timer.elapsed();
    std::cout << "decoding time (wall): " << elapsed.wall / 1e9 << std::endl;

    py::list py_prefix;
    for (int k = 0; k < best_prefix.size(); k++)
    {
        py_prefix.append(best_prefix[k]);
    }
    return py::make_tuple(py_prefix, best_score);
}

BOOST_PYTHON_MODULE(fastdecode)
{
    np::array::set_module_and_type("numpy", "ndarray");

    py::def("decode_lm_wrapper", &decode_lm_wrapper);
}
