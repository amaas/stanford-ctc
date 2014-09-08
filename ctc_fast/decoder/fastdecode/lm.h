#pragma once
#include <string>
#include <math.h>
#include <fstream>
#include <boost/unordered_map.hpp>
#include <iostream>

namespace LM_CONSTS
{
    const std::string UNK = "<UNK>";
    const std::string START = "<s>";
    const std::string END = "</s>";
    // Scale from log10 to ln
    const float SCALE = log(10);
};

class LM
{
  public:

    LM(const std::string& fname)
    {
        std::ifstream ifs(fname.c_str());

        std::cout << "Reading header" << std::endl;
        read_header(ifs);
        std::cout << "Reading unigrams" << std::endl;
        load_ug(ifs);
        std::cout << "Reading bigrams" << std::endl;
        load_bg(ifs);
        std::cout << "Done." << std::endl;
        // TODO Handle trigrams

        start = word_to_int[LM_CONSTS::START];
        end = word_to_int[LM_CONSTS::END];
        unk = word_to_int[LM_CONSTS::UNK];
    }

    void load_chars(const std::string& fname);
    void load_words(const std::string& fname);
    int get_word_id(const std::string& word_string);

    void read_header(std::ifstream& ifs);
    void load_ug(std::ifstream& ifs);
    void load_bg(std::ifstream& ifs);

    float ug_prob(int wid);
    float bg_prob(int w1, int w2);

    float score_bg(const std::string& sentence);

  //protected:

    boost::unordered_map<std::string, int> word_to_int;
    boost::unordered_map<int, std::pair<float, float> > ug_probs;
    boost::unordered_map<std::pair<int, int>, float> bg_probs;
    boost::unordered_map<std::string, int> char_map;

    std::vector<std::string> word_list;
    int start, end, unk;
    int num_words;
    bool is_trigram;
};
