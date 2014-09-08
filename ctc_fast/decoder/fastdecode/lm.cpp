#include "lm.h"
#include <fstream>
#include <assert.h>
#include <iostream>
#include <boost/algorithm/string.hpp>

void LM::load_chars(const std::string& fname)
{
    std::string line;
    std::ifstream ifs(fname.c_str());
    std::vector<std::string> splits;
    while (std::getline(ifs, line))
    {
        boost::trim(line);
        boost::split(splits, line, boost::is_any_of(" \t\n"));
        char_map[splits[0]] = atoi(splits[1].c_str());
        std::cout << splits[0] << " -> " << char_map[splits[0]] << std::endl;
    }
    ifs.close();
}

void LM::load_words(const std::string& fname)
{
    std::string line;
    std::ifstream ifs(fname.c_str());
    while (std::getline(ifs, line))
    {
        word_list.push_back(boost::trim_copy(line));
    }
    std::cout << "Loaded " << word_list.size() << " words" << std::endl;
}

int LM::get_word_id(const std::string& word_string)
{
    if (word_to_int.find(word_string) != word_to_int.end())
        return word_to_int[word_string];
    else
        return unk;
}

void LM::read_header(std::ifstream& ifs)
{
    std::string line = "";
    std::vector<std::string> splits;
    while (boost::trim_copy(line) != "\\data\\")
        std::getline(ifs, line);
    std::getline(ifs, line);
    assert(line.find("ngram 1") != std::string::npos);
    boost::trim(line);
    boost::split(splits, line, boost::is_any_of("="));
    num_words = atoi(splits[1].c_str());
    std::getline(ifs, line);
    assert(line.find("ngram 2") != std::string::npos);
    std::getline(ifs, line);
    if (line.find("ngram 3") != std::string::npos)
        is_trigram = true;
    else
        is_trigram = false;
}

void LM::load_ug(std::ifstream& ifs)
{
    int count = 0;
    std::string line;
    std::vector<std::string> splits;
    while (line.find("\\1-grams") == std::string::npos)
        std::getline(ifs, line);
    while (true)
    {
        std::getline(ifs, line);
        //std::cout << line << std::endl;
        boost::trim(line);
        boost::split(splits, line, boost::is_any_of(" \t"));
        if (splits.size() < 2)
            break;
        word_to_int[splits[1]] = count;
        if (splits.size() == 3)
        {
            ug_probs[count] = std::make_pair(LM_CONSTS::SCALE * atof(splits[0].c_str()),
                                             LM_CONSTS::SCALE * atof(splits[2].c_str()));
        }
        else
        {
            ug_probs[count] = std::make_pair(LM_CONSTS::SCALE * atof(splits[0].c_str()),
                                             0.0f);
        }
        count += 1;
    }
    std::cout << "Loaded " << count << " unigrams" << std::endl;
}

void LM::load_bg(std::ifstream& ifs)
{
    std::string line;
    std::vector<std::string> splits;
    while (line.find("\\2-grams") == std::string::npos)
        std::getline(ifs, line);
    int count = 0;
    while (true)
    {
        std::getline(ifs, line);
        boost::trim(line);
        boost::split(splits, line, boost::is_any_of(" \t"));
        if (splits.size() < 3 || splits[0] == "\\end\\")
            break;
        // TODO Handle trigrams backoffs here
        bg_probs[std::make_pair(word_to_int[splits[1]],
                word_to_int[splits[2]])] = LM_CONSTS::SCALE * atof(splits[0].c_str());
        count += 1;
    }
    std::cout << "Loaded " << count << " bigrams" << std::endl;
}

float LM::ug_prob(int wid)
{
    return ug_probs[wid].first;
}

float LM::bg_prob(int w1, int w2)
{
    float p = bg_probs[std::make_pair(w1, w2)];
    if (p == 0.0)
    {
        p += ug_probs[w1].second + ug_probs[w2].first;
    }
    return p;
}

float LM::score_bg(const std::string& sentence)
{
    std::vector<std::string> splits;
    std::string s = boost::trim_copy(sentence);
    boost::split(splits, s, boost::is_any_of(" \t"));
    std::vector<int> word_ids;
    for (int k = 0; k < splits.size(); k++)
    {
        word_ids.push_back(get_word_id(splits[k]));
    }

    float score = 0.0;
    std::cout << bg_prob(start, word_ids[0]) << std::endl;
    score += bg_prob(start, word_ids[0]);
    for (int k = 0; k < word_ids.size() - 1; k++)
    {
        std::cout << k << " " << bg_prob(word_ids[k], word_ids[k+1]) << std::endl;
        score += bg_prob(word_ids[k], word_ids[k+1]);
    }

    std::cout << bg_prob(word_ids[word_ids.size() - 1], end) << std::endl;
    score += bg_prob(word_ids[word_ids.size() - 1], end);
    return score;
}
