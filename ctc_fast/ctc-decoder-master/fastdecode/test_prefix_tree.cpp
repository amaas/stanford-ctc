#include "prefix_tree.h"

int main(int argc, char** argv)
{
    // TODO FIXME PARAM Import
    std::string CHARMAP_PATH = "/scail/group/deeplearning/speech/zxie/kaldi-stanford/kaldi-trunk/egs/wsj/s6/ctc-utils";

    PrefixTree ptree(CHARMAP_PATH + "/chars.txt",
                     CHARMAP_PATH + "/wordlist",
                     CHARMAP_PATH + "/lm_bg.arpa");

    std::cout << "Added " << ptree.path_count << " prefix paths" << std::endl;
}
