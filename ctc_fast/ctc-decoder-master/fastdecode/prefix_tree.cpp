#include "prefix_tree.h"

void PrefixTree::add_path(const std::string& prefix, PrefixTreeNode::Ptr node, int word_id)
{
    path_count++;

    std::string p = prefix.substr(0, 1);
    std::string rest = prefix.substr(1);

    PrefixTreeNode::Ptr next(new PrefixTreeNode);
    next->word_id = lm.unk;
    if (node->children.find(lm.char_map[p]) == node->children.end())
        node->children[lm.char_map[p]] = next;
    else
        next = node->children[lm.char_map[p]];

    if (rest.length() == 0)
    {
        next->is_word = true;
        next->word_id = word_id;
        return;
    }
    else
    {
        next->is_prefix = true;
        add_path(rest, next, word_id);
    }
}
