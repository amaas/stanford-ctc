#pragma once
#include "prefix_tree.h"
#include <limits>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/numeric.hpp>

namespace py = boost::python;
namespace np = boost::python::numeric;

const double EPS = 1e-300;
const double NEG_INF = -1 * std::numeric_limits<double>::infinity();
// FIXME Add beta as parameter once move everything to a class
const double COMP_BETA = 0.0;

struct PrefixHyp
{
    PrefixHyp() : nw(0), p_b(NEG_INF), p_nb(NEG_INF), score(NEG_INF) {}
    int nw;
    int wid;
    double p_b;
    double p_nb;
    double score;
    PrefixTreeNode::Ptr node;
    typedef boost::shared_ptr<PrefixHyp> Ptr;
};

typedef std::vector<int> PrefixKey;
typedef boost::unordered_map<PrefixKey, PrefixHyp::Ptr> PrefixMap;
typedef std::pair<PrefixKey, PrefixHyp::Ptr> PrefixPair;

double combine(double a, double b, double c=NEG_INF);
double prefix_hyp_score(PrefixHyp::Ptr hyp);
// TODO Pass ref?
bool comp_prefix_pair(const PrefixPair& x, const PrefixPair& y);

double decode_bg_lm(double* probs, int N, int T, PrefixTree& ptree,
        int beam, double alpha, double beta, PrefixKey& best_prefix);

py::tuple decode_lm_wrapper(np::array probs, int beam, double alpha, double beta);
