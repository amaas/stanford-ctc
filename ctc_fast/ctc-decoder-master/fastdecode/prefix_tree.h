#pragma once
#include "lm.h"
#include <iostream>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
//#include <boost/progress.hpp>

// TODO FIXME PARAM Import these parameters in using boost.python
namespace PREFIX_TREE_CONSTS
{
    const std::string SPACE = "<SPACE>";
    const std::string SPECIALS_LIST[2] = {"<SPACE>", "<NOISE>"};
};

struct PrefixTreeNode
{
    PrefixTreeNode() : is_prefix(false), is_word(false) {}
    bool is_prefix;
    bool is_word;
    int word_id;
    typedef boost::shared_ptr<PrefixTreeNode> Ptr;
    boost::unordered_map<int, Ptr> children;
};

class PrefixTree
{
  public:
    PrefixTree(const std::string& char_map_file,
               const std::string& word_list_file,
               const std::string& arpa_lm_file) :
        lm(arpa_lm_file), path_count(0), root(new PrefixTreeNode)
    {
        lm.load_chars(char_map_file);
        lm.load_words(word_list_file);

        root->is_prefix = true;
        space = lm.char_map[PREFIX_TREE_CONSTS::SPACE];

        BOOST_FOREACH (std::string w, PREFIX_TREE_CONSTS::SPECIALS_LIST)
        {
            PrefixTreeNode::Ptr n(new PrefixTreeNode());
            n->is_word = true;
            n->word_id = lm.get_word_id(w);
            root->children.insert(std::make_pair(lm.char_map[w], n));
        }

        //boost::progress_display show_progress(lm.word_list.size());
        BOOST_FOREACH(std::string word, lm.word_list)
        {
            add_path(word, root, lm.get_word_id(word));
            //++show_progress;
        }
    }

    void add_path(const std::string& prefix, PrefixTreeNode::Ptr node, int word_id);
    int path_count;

  //protected:
    LM lm;
    PrefixTreeNode::Ptr root;
    int space;
};
