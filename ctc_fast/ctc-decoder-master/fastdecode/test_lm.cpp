#include "lm.h"
#include <iostream>

int main(int argc, char** argv)
{
    LM lm("/afs/cs.stanford.edu/u/awni/wsj/ctc-utils/lm_bg.arpa");
    std::cout << lm.word_to_int["ACCEPT"] << std::endl;
    std::cout << lm.score_bg("HELLO AGAIN") << std::endl;
    std::cout << lm.score_bg("HELLO ABDO") << std::endl;
}
